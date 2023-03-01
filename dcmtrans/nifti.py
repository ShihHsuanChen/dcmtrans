r"""
require package nibabel
"""
import os
from typing import Iterable, Union
from collections import namedtuple
import numpy as np
import nibabel as nib

from .typing import NTuple, PathLike
from .reconstruction import RecInfo
from .volume import build_volume_from_recon_info


VolumeCoor = namedtuple(
    'VolumeCoor',
    ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing']
)
r"""
Not used now
- PixelSpacing: (tuple) spacing between pixels (h,w) or (j,i), same as PixelSpacing tag
- ImageOrientationPatient: [<float>]*6
- ImagePositionPatient: [<float>]*3
"""


def recinfo2nii(rec_info: RecInfo[PathLike], **kwargs) -> nib.nifti1.Nifti1Image:
    vol = build_volume_from_recon_info(rec_info)
    return dcmvol2nii(vol, rec_info, **kwargs)


def dcmvol2nii(
        volume: np.ndarray,
        rec_info: RecInfo,
        **kwargs,
        ) -> nib.nifti1.Nifti1Image:
    nibobj = dcmvol2nii_3dslicer(
        volume,
        rec_info.ImagePositionPatient,
        rec_info.ImageOrientationPatient,
        rec_info.PixelSpacing,
    )
    return nibobj


def dcmvol2nii_3dslicer(
        vol: np.ndarray,
        ipp: Iterable[Union[np.ndarray, NTuple(float, 3)]],
        iop: Union[Iterable, np.ndarray, NTuple(float, 6)],
        ps: Union[Iterable, np.ndarray, NTuple(float, 2)],
        ) -> nib.nifti1.Nifti1Image:
    r"""
    Argument:
    - vol: (np.ndarray) volume array in shape (D,H,W)
    - ipp: (iterable) collection of ImagePositionPatient. Shape (D,3)
    - iop: (iterable) collection of ImageOrientationPatient. Shape (D,6) or (6,)
    - ps:  (iterable) collection of PixelSpacing. Shape (D,2) or (2,)
    where D is the number of DICOM slices in the series.

    Returns:
    Nifti1Image object

    -----------
    How:
    - volume

        1. Sort index D into Dicom standard patient coordinate

            * feet  -> head
            * front -> back
            * right -> left

        2. Sort index H into Dicom standard patient coordinate

            * head  -> feet
            * front -> back

        3. Sort index W into Dicom standard patient coordinate

            * right -> left
            * front -> back

        4. If index changed, re-calculate the ImagePositionPatient (position of the top left corner
           of the first slice, or index [0,0,0]) and ImageOrientPatient

    - spacing

        Calculate spacing along Sd=(Xd,Yd,Zd), where subscript d indicate Dicom standard patient coordinate.
        For example,
        * a normal    axial view series -> ($_PS[0], $_PS[1], $ST    )
        * a normal  coronal view series -> ($_PS[1], $ST    , $_PS[0])
        * a normal sagittal view series -> ($ST    , $PS[0] , $_PS[1])
        where $SL is slice thickness, $_PS = $PS[::-1] since PixelSpacing is spacing along (h,w)

    - affine matrix

        1. calculate cross product of $IOP[0:3] and $IOP[3:6] as $CP
            $CP = [
                $IOP[1]*$IOP[5],
                $IOP[2]*$IOP[3],
                $IOP[0]*$IOP[4],
            ]
        2. then affine matrix =
            [[-$IOP[0]*$_PS[0]], [-$IOP[3]*$_PS[1]], [-$CP[0]*$ST], [-$IPP[0]]],
            [[-$IOP[1]*$_PS[0]], [-$IOP[4]*$_PS[1]], [-$CP[1]*$ST], [-$IPP[1]]],
            [[ $IOP[2]*$_PS[0]], [ $IOP[5]*$_PS[1]], [ $CP[2]*$ST], [ $IPP[2]]],
            [[        0       ], [        0       ], [       0   ], [    1   ]],

    - pixdim
        [1, $_PS[0], $_PS[1], $ST, 0, 0, 0]
    """
    # convert to np.array
    vol = vol.copy()
    ipp, iop, ps = np.array(ipp), np.array(iop), np.array(ps)

    # check dimension
    print(vol.shape)
    D,H,W = vol.shape
    if ps.ndim == 2:
        ps = ps[0]
    if iop.ndim == 2:
        iop = iop[0]
    assert ipp.ndim == 2
    assert iop.ndim == 1
    assert  ps.ndim == 1
    assert ipp.shape[0] == D
    assert ipp.shape[1] == 3
    assert iop.shape[0] == 6
    assert  ps.shape[0] == 2
    _ps = ps[::-1]

    # check orientation
    # get main axis of D
    dv = ipp[1]-ipp[0]
    _iop = np.array([iop[:3], iop[3:]])

    axis_d = np.argmax(np.abs(dv))
    axis_wh = np.argmax(np.abs(_iop), axis=1)
    direction = np.sign(dv[axis_d])

    if direction < 0:
        vol = np.flip(vol, 0)
        ipp = np.flip(ipp, 0)

    for i, npx in enumerate([W,H]): # loop for w,h
        if i == 1 and axis_wh[i] == 2: # H~Z
            if _iop[i, axis_wh[i]] < 0: # H~Z, head -> feet
                continue
        else:
            if _iop[i, axis_wh[i]] > 0: # H~Z, head -> feet
                continue
        # flip
        vol = np.flip(vol, i)
        ipp += _iop[i].reshape(1,-1)*_ps[i]*(npx-1)
        _iop[i] = -_iop[i]

    # calculate affine matrix and pixdim
    cp = np.cross(*_iop) # cross product
    st = np.sqrt(np.sum(dv**2)) # slice thickness
    _ipp = ipp[0]
    affine = np.array([
        [-_iop[0,0]*_ps[0], -_iop[1,0]*_ps[1], -cp[0]*st, -_ipp[0]],
        [-_iop[0,1]*_ps[0], -_iop[1,1]*_ps[1], -cp[1]*st, -_ipp[1]],
        [ _iop[0,2]*_ps[0],  _iop[1,2]*_ps[1],  cp[2]*st,  _ipp[2]],
        [                0,                 0,         0,        1],
    ])
    # pixdim = np.array([1, _ps[0], _ps[1], st, 0, 0, 0])
    nibobj = nib.Nifti1Image(vol.T, affine)
    nibobj.header.set_sform(None, code=1)
    nibobj.header.set_qform(None, code=1)
    return nibobj


def nii2dcmvol(nibobj: nib.nifti1.Nifti1Image):
    return np.array(nibobj.dataobj).T


def create_affine(
        ipp: Iterable,
        iop: Iterable,
        ps: Iterable,
        ) -> NTuple(np.ndarray, 2):
    """ obtain from https://gist.github.com/dgobbi/ab71f5128aa43f0d33a41775cb2bcca6
    Generate a NIFTI affine matrix from DICOM IPP and IOP attributes.
    
    Argument:
    - ipp: (iterable) collection of ImagePositionPatient. Shape (N,3)
    - iop: (iterable) collection of ImageOrientationPatient. Shape (N,6)
    - ps:  (iterable) collection of PixelSpacing. Shape (N,2)
    
    , where N is the number of DICOM slices in the series.
    
    The return values are the NIFTI affine matrix and the NIFTI pixdim.
    Note the the output will use DICOM anatomical coordinates:
    x increases towards the left, y increases towards the back.
    """
    # solve Ax = b where x is slope, intecept
    ipp, iop, ps = np.array(ipp), np.array(iop), np.array(ps)
    n = ipp.shape[0]
    A = np.column_stack([np.arange(n), np.ones(n)])
    x, r, rank, s = np.linalg.lstsq(A, ipp, rcond=None)
    # round small values to zero
    x[(np.abs(x) < 1e-6)] = 0.0
    vec = x[0,:] # slope
    pos = x[1,:] # intercept

    # pixel spacing should be the same for all image
    spacing = np.ones(3)
    spacing[0:2] = ps[0,::-1] # reverse since PixelSpacing is (h,w)
    if np.sum(np.abs(ps - spacing[0:2])) > spacing[0]*1e-6:
        sys.stderr.write("Pixel spacing is inconsistent!\n");

    # compute slice spacing
    spacing[2] = np.round(np.sqrt(np.sum(np.square(vec))), 7)

    # get the orientation
    iop_average = np.mean(iop, axis=0)
    u = iop_average[0:3]
    u /= np.sqrt(np.sum(np.square(u)))
    v = iop_average[3:6]
    v /= np.sqrt(np.sum(np.square(v)))

    # round small values to zero
    u[(np.abs(u) < 1e-6)] = 0.0
    v[(np.abs(v) < 1e-6)] = 0.0

    # create the matrix
    mat = np.eye(4)
    mat[0:3,0] = u*spacing[0]
    mat[0:3,1] = v*spacing[1]
    mat[0:3,2] = vec
    mat[0:3,3] = pos

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot(vec, np.cross(u, v))
    qfac = np.sign(dv)
    if np.abs(qfac*dv - spacing[2]) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n");

    # compute the nifti pixdim array
    pixdim = np.hstack([np.array(qfac), spacing])
    return mat, pixdim


def convert_coords(vol: np.ndarray, mat: np.ndarray, inplace: bool = False) -> NTuple(np.ndarray, 2):
    """ obtain from https://gist.github.com/dgobbi/ab71f5128aa43f0d33a41775cb2bcca6
    Convert a volume from DICOM coords to NIFTI coords or vice-versa.
    For DICOM, x increases from right to the left and y increases to the back of patient.
    For NIFTI, x increases form left to the right and y increases to the front of patient.
    The conversion is done in-place (volume and matrix are modified).

    Arguments:
    - vol: (np.ndarray) Volume. Shape (Z,Y,X) of Dicom patient coordinate system
    - mat: (np.ndarray) Affine matrix from create_affine output
    - inplace: (bool) apply inplace operation. (default: False)

    Returns:
    - new_vol: (np.ndarray) Volume in standard Nifti coords. Shape (Z,Y,X) of Nifti1 patient coordinate system
    - new_mat: (np.ndarray) Affine matrix corresponding to new volume

    Note that:
    1. This is only for axial data.
    2. NiftiImage header fields: sform_code=2, qform_code=0 (method 1)
    """
    # the x direction and y direction are flipped
    if not inplace:
        vol = vol.copy()
        mat = mat.copy()
    convmat = np.eye(4)
    convmat[0,0] = -1.0
    convmat[1,1] = -1.0

    # apply the coordinate change to the matrix
    mat[:] = np.dot(convmat, mat)

    # look for x and y elements with greatest magnitude
    xabs = np.abs(mat[:,0])
    yabs = np.abs(mat[:,1])
    xmaxi = np.argmax(xabs)
    yabs[xmaxi] = 0.0
    ymaxi = np.argmax(yabs)

    # re-order the data to ensure these elements aren't negative
    # (this may impact the way that the image is displayed, if the
    # software that displays the image ignores the matrix).
    if mat[xmaxi,0] < 0.0:
        # flip x
        vol[:] = np.flip(vol, 2)
        mat[:,3] += mat[:,0]*(vol.shape[2] - 1)
        mat[:,0] = -mat[:,0]
    if mat[ymaxi,1] < 0.0:
        # flip y
        vol[:] = np.flip(vol, 1)
        mat[:,3] += mat[:,1]*(vol.shape[1] - 1)
        mat[:,1] = -mat[:,1]

    # eliminate "-0.0" (negative zero) in the matrix
    mat[mat == 0.0] = 0.0
    return vol, mat
