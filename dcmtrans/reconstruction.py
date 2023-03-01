import warnings
from typing import Optional, Generic, List, Set, Union, Dict, Any
from pydicom import FileDataset
from pydantic.generics import GenericModel

from .typing import T, NTuple


__all__ = ['RecInfo', 'reconstruct_series', 'get_instance_info']


class RecInfo(GenericModel, Generic[T]):
    r"""
    Usage:
        - RecInfo(...): any key type
        - RecInfo[int](...): restrict to integer key type
        - RecInfo[MyType](...): restrict to "MyType" key type
    Attributes
        - index_list: [<InstanceNumber;int>]
        - index_map: {<InstanceNumber;int>: <key of dicom_dict>}
        - index_dicom_dict: {<InstanceNumber;int>: <FileDataset>}
        - series_like: <series like axial/coronal/sagittal/other>
        - chirality: <+1/-1> +1 if right-handed, index increasing from
                     anterior(front)/right/feet to posterior(back)/left/head
                     or ((Pn - Pn-1) dot k) > 0; vice versa
                     Return `None` if series is not a 3D volume
        - use_contrast: <True: w/ contrast; False: w/o contrast>
        - slice_spacing: <calculated spacing; float, optional>
        - description: (str) SeriesDescription 
        - N_pixel_i: (int) number of pixels along first row, or W, i 
        - N_pixel_j: (int) number of pixels along first column, or H, j 
        - spacing_i: (float, optional) spacing between the centers of adjacent
                     columns, or horizontal spacing in mm
        - spacing_j: (float, optional) spacing between the centers of adjacent
                     rows, or vertical spacing in mm
        - spacing_slice: (set, float, optional) spacing between slices in mm 
        - slice_thickness: (float, optional) slice thickness in mm
        - PixelSpacing: (tuple) spacing between pixels (H,W) or (j,i), same as PixelSpacing tag
        - ImageOrientationPatient: [<float>]*6, optional
        - ImagePositionPatient: [<float>]*3, optional
        - PatientPosition: (str, optional) PatientPosition
    """
    index_list: List[int]
    r""" list of InstanceNumber (int) """
    index_map: Dict[int, T]
    r""" dictionary: InstanceNumber(int) -> key of dicom_dict """
    index_dicom_dict: Dict[int, Any]
    r""" dictionary: InstanceNumber(int) -> value of dicom_dict """
    series_like: Optional[str]
    r""" series like axial/coronal/sagittal/other """
    chirality: Optional[int]
    r""" <+1/-1>
    +1 if right-handed, index increasing from
    anterior(front)/right/feet to posterior(back)/left/head
    or ((Pn - Pn-1) dot k) > 0; vice versa
    Return `None` if series is not a 3D volume
    """
    use_contrast: bool
    r""" check if ContrastBolusAgent has value.
    True: w/ contrast; False: w/o contrast
    """
    description: str = ''
    r""" SeriesDescription """
    N_pixel_i: int
    r""" number of pixels along first row, or W, i """
    N_pixel_j: int
    r""" number of pixels along first column, or H, j """
    spacing_i: Optional[float]
    r""" spacing between the centers of adjacent columns, or horizontal spacing in mm """
    spacing_j: Optional[float]
    r""" spacing between the centers of adjacent rows, or vertical spacing in mm """
    spacing_slice: Union[None, Set[float], float] = None
    r""" spacing between slices in mm """
    slice_thickness: Optional[float] = None
    r""" slice thickness """
    PixelSpacing: NTuple(float, 2)
    r""" spacing between pixels (H,W) or (j,i), same as dicom records """
    ImageOrientationPatient: Optional[NTuple(float, 6)] = None
    r""" ImageOrientationPatient [<float>]*6, optional """
    ImagePositionPatient: Optional[List[NTuple(float, 3)]] = None
    r""" ImagePositionPatient [<float>]*3, optional """
    PatientPosition: Optional[str] = None
    r""" PatientPosition """


def classify(dcmObj):
    return getattr(dcmObj, 'Modality', None)


def reconstruct_series(
        dicom_dict: Dict[T, FileDataset],
        modality: Optional[str] = None,
        ) -> RecInfo:
    '''
    1. get InstanceNumber as the original indexing and make map indexing -> dicom filename
    2. check inner product of ImageOrientationPatient of first and second images: if == 0 -> discard first
    3. check alignment from ImagePositionPatient 
    4. calculate spacing
    5. retag series with ImageOrientationPatient: Axial-like, Coronal-like, Sagittal-like, Other
    6. retag contrast with ContrastBolusAgent: null -> w/o contrast, otherwise -> w/
    7. return map, dicom_dict that indexing by InstanceNumber, use_contrast, series_like, ..

    Input: dicom_dict
        dicom_dict: {<unique key(instance_no or filename)>: <FileDataset object of same SeriesDescription>}
    Return: RecInfo
        RecInfo:
        * index_list: [<InstanceNumber;int>]
        * index_map: {<InstanceNumber;int>: <key of dicom_dict>}
        * index_dicom_dict: {<InstanceNumber;int>: <FileDataset>}
        * series_like: <series like axial/coronal/sagittal/other>
        * chirality: <+1/-1> +1 if right-handed, index increasing from
                     anterior(front)/right/feet to posterior(back)/left/head
                     or ((Pn - Pn-1) dot k) > 0; vice versa
                     Return `None` if series is not a 3D volume
        * use_contrast: <True: w/ contrast; False: w/o contrast>
        * slice_spacing: <calculated spacing; float, optional>
        * description: (str) SeriesDescription 
        * N_pixel_i: (int) number of pixels along first row, or W, i 
        * N_pixel_j: (int) number of pixels along first column, or H, j 
        * spacing_i: (float, optional) spacing between the centers of adjacent
        *            columns, or horizontal spacing in mm
        * spacing_j: (float, optional) spacing between the centers of adjacent
        *            rows, or vertical spacing in mm
        * spacing_slice: (float, optional) spacing between slices in mm 
        * slice_thickness: (float, optional) slice thickness 
        * PixelSpacing: (tuple) spacing between pixels (H,W) or (j,i), same as PixelSpacing tag
        * ImageOrientationPatient: [<float>]*6, optional
        * ImagePositionPatient: [<float>]*3, optional
        * PatientPosition: (str, optional) PatientPosition
    '''

    # 1. get InstanceNumber as the original indexing and make map indexing -> dicom filename
    index_map = {
        int(x.InstanceNumber): unikey
        for unikey, x in dicom_dict.items()
        if getattr(x, 'InstanceNumber', None) is not None
    }

    index_data_dict = {
        i: dicom_dict.get(unikey)
        for i, unikey in index_map.items()
    }

    index_list = list(index_map.keys())
    index_list.sort()

    if modality is None:
        mod_list = list({classify(x) for _, x in dicom_dict.items()})
        if len(mod_list) > 1:
            raise AssertionError('Multiple Modality were found')
        mod = mod_list[0]
    else:
        mod = modality.upper()

    if mod == 'CR':
        if len(index_list) == 0:
            raise AssertionError('No instance was found')
        elif len(index_list) > 1:
            raise AssertionError('Modality "CR": Multiple instances were found.')

        dcmObj = index_data_dict.get(index_list[0])
        chirality = None
        spacing_slice = None
        series_like = getattr(dcmObj, 'ViewPosition', None)
        ipp = None
        iop = None
    elif mod == 'DX':
        if len(index_list) == 0:
            raise AssertionError('No instance was found')
        elif len(index_list) > 1:
            raise AssertionError('Modality "DX": Multiple instances were found.')

        dcmObj = index_data_dict.get(index_list[0])
        chirality = None
        spacing_slice = None
        series_like = None # AP / PA / ???
        ipp = None
        iop = None
    elif mod == 'CT':
        if len(index_list) < 2:
            raise AssertionError('Cannot be reconstructed: Number of CT images less than 2')
        res = _recon_ct(index_map, index_data_dict, index_list)
        index_data_dict = res['index_data_dict']
        index_list = res['index_list']
        index_map = res['index_map']
        chirality = res['chirality']
        spacing_slice = res['spacing_slice']
        series_like = res['series_like']
        ipp = res['image_position_patient']
        iop = res['image_orientation_patient']
    # elif mod == 'MR':
        # TODO
    else:
        raise NotImplementedError(f'No reconstruction methods for {mod}')

    # 6. retag contrast with ContrastBolusAgent: null -> w/o contrast, otherwise -> w/
    use_contrast = getattr(index_data_dict.get(index_list[0]), 'ContrastBolusAgent', False)
    use_contrast = (use_contrast not in [None, '', False])
    
    # 7. return map, dicom_dict that indexing by InstanceNumber, use_contrast, series_like, ..
    first = index_data_dict.get(index_list[0])
    index_dicom_dict = {i: dicom_dict.get(unikey) for i, unikey in index_map.items()}
    rec_info = RecInfo(
        index_list=index_list,
        index_map=index_map,
        index_dicom_dict=index_dicom_dict,
        chirality=chirality,
        series_like=series_like,
        use_contrast=use_contrast,
        description=getattr(first, 'SeriesDescription', ''),
        N_pixel_i=getattr(first, 'Rows', -1),
        N_pixel_j=getattr(first, 'Columns', -1),
        # i -> spacing between columns
        spacing_i=float(getattr(first, 'PixelSpacing', [1, 1])[1]),
        # j -> spacing between rows
        spacing_j=float(getattr(first, 'PixelSpacing', [1, 1])[0]),
        spacing_slice=spacing_slice,
        slice_thickness=getattr(first, 'SliceThickness', None),
        PixelSpacing=tuple(getattr(first, 'PixelSpacing', [1, 1])),
        PatientPosition=getattr(first, 'PatientPosition', None),
        ImagePositionPatient=ipp,
        ImageOrientationPatient=iop,
    )
    return rec_info


def _recon_ct(index_map, index_data_dict, index_list):
    # for CT
    # 2. check inner product of ImageOrientationPatient of first and second images: if == 0 -> discard first
    #    2.1 get first and second
    dcm1 = index_data_dict.get(index_list[0])
    dcm2 = index_data_dict.get(index_list[1])
    #    2.2 get ImageOrientationPatient and get inner product
    iop1 = getattr(dcm1, 'ImageOrientationPatient', None)
    if iop1 is None:
        raise AssertionError('ImageOrientationPatient not found')
    iop2 = getattr(dcm2, 'ImageOrientationPatient', None)
    if iop2 is None:
        raise AssertionError('ImageOrientationPatient not found')
    iop1 = [float(x) for x in iop1]
    iop2 = [float(x) for x in iop2]
    dot = (  (sum([iop1[i]*iop2[i] for i in range(3)]) * sum([iop1[i+3]*iop2[i+3] for i in range(3)]))
           - (sum([iop1[i]*iop2[i+3] for i in range(3)]) * sum([iop1[i+3]*iop2[i] for i in range(3)])))
    if abs(dot) < 0.1:
        # discard first
        index_map = {k:v for k, v in index_map.items() if k != index_list[0]}
        index_data_dict = {k:v for k, v in index_data_dict.items() if k != index_list[0]}
        index_list = list(index_map.keys())
        index_list.sort()

    # 3. check alignment from ImagePositionPatient 
    #    3.1 check image size
    if len(set([x.Columns for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('Columns of images mismatch')
    if len(set([x.Rows for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('Rows of images mismatch')
    #    3.2 check pixel spacing in precision of 0.001
    if len(set([round(float(x.PixelSpacing[0])/1E-3) for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('PixelSpacing of columns mismatch')
    if len(set([round(float(x.PixelSpacing[1])/1E-3) for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('PixelSpacing of columns mismatch')
    #    3.2 check slice spacing in precision of 0.01
    okay = True
    spacing_list = list()
    iop_list = [[float(x)
                 for x in index_data_dict.get(ind).ImageOrientationPatient]
                for ind in index_list]
    ipp_list = [[float(x)
                 for x in index_data_dict.get(ind).ImagePositionPatient]
                for ind in index_list]
    chirality_list = list()
    msglist = list()
    for i, ind1 in enumerate(index_list):
        if i == 0:
            continue

        ind0 = index_list[i-1]
        if (ind1 - ind0) != 1:
            okay = False
            msglist.append(f'index jump {ind0} -> {ind1}')
            continue
        iop = iop_list[i-1]
        pos1 = ipp_list[i]
        pos0 = ipp_list[i-1]
        dp = [pos1[j] - pos0[j] for j in range(3)]
        spacing = sum([x**2 for x in dp])**0.5
        if abs(spacing) < 1e-4:
            warnings.warn(f'spacing between {ind0} and {ind1} is 0')
            continue
        tmp_dot = sum([
            dp[0] *iop[1]*iop[3+2],
            dp[1] *iop[2]*iop[3+0],
            dp[2] *iop[0]*iop[3+1],
            -dp[0]*iop[2]*iop[3+1],
            -dp[1]*iop[0]*iop[3+2],
            -dp[2]*iop[1]*iop[3+0]
            ])
        if abs(tmp_dot/spacing) < 0.99:
            msglist.append(f'{ind0} and {ind1} do not aligned')
            okay = False
            continue
        chirality_list.append(tmp_dot/spacing>0)
        spacing_list.append(spacing)

    if not okay:
        raise AssertionError('\n'.join(msglist))
    if len(set(chirality_list)) > 1:
        raise AssertionError('Direction (Chirality) of images mismatch')
    elif len(chirality_list) == 0:
        chirality = None
    else:
        chirality = chirality_list[0]
    chirality = 1 if chirality else -1

    # 4. calculate spacing
    # if len(set([round(x*1E3) for x in spacing_list])) != 1:
        # raise AssertionError('Slice Spacing mismatch')
    # spacing = abs(spacing_list[0])
    # 5. retag series with ImageOrientationPatient: Axial-like, Coronal-like, Sagittal-like, Other
    iop = iop_list[0]
    k_like = {'axial': [0,0,1], 'coronal': [0,1,0], 'sagittal': [1,0,0]}
    series_like = 'other'
    for cls, vec in k_like.items():
        if (abs(sum([iop[i]  *vec[i] for i in range(3)])) < 0.1 and
            abs(sum([iop[i+3]*vec[i] for i in range(3)])) < 0.1):
            series_like = cls
            break

    return {'index_data_dict': index_data_dict,
            'index_list': index_list,
            'index_map': index_map,
            'chirality': chirality,
            # 'spacing_slice': float('%.3f' % spacing),
            'spacing_slice': set([round(x*1e4)*1e-4 for x in spacing_list]),
            'series_like': series_like,
            'image_orientation_patient': iop,
            'image_position_patient': ipp_list,
            }


def get_instance_info(dcmObj):
    STORE_VAR = ['Modality', 'PatientID', 'AccessionNumber',
                 'StudyDate', 'StudyTime', 'StudyDescription',
                 'SeriesDate', 'SeriesTime', 'SeriesDescription',
                 'SamplesPerPixel', 'Rows', 'Columns', 'PixelSpacing',
                 'SliceThickness', 'SpacingBetweenSlices', 'SliceLocation',
                 'PhotometricInterpretation',
                 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                 'PlanarConfiguration', 'PixelAspectRatio', 'SmallestImagePixelValue',
                 'LargestImagePixelValue', 'ColorSpace', 'BodyPartExamined',
                 'PatientPosition', 'ImageOrientationPatient', 'ImagePositionPatient',
                 'PatientOrientation', 'AnatomicalOrientationType', 'ViewPosition'
                 ]
    res = dict()
    res['bits_origin'] = getattr(dcmObj, 'BitsStored', None)
    image_position = getattr(dcmObj, 'ImagePositionPatient', None)
    if image_position is not None:
        image_position = [float('%.4f'%float(x)) for x in image_position]
    res['image_position'] = image_position
    return res
