import numpy as np
import pydicom

from .base import BaseTransManager
from .trans_method import do_nothing


__all__ = [
    'pi_trans',
    'pi_classifier',
    'pi_monochrome1',
    'pi_monochrome2',
    'pi_rgb',
    'pi_ybr_full',
    'pi_ybr_full_422',
    'pi_ybr_partial_420',
    'pi_ybr_ict',
    'pi_ybr_rct',
    'get_pi_trans_func',
]

class PITransManager(BaseTransManager):
    def get_mode(self, dcmObj: pydicom.FileDataset) -> str:
        """
        Get photometric interpretation (pi) transform method name from dicom object

        :param dcmObj: object read from pydicom
        :return: pi transform method name. return value can be one of follows:
                 MONOCHROME1, MONOCHROME2, PALETTE COLOR, RGB, ...
                 Details can be found here https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004.
                 However this module can only deal with class of MONOCHROME1, and MONOCHROME2
        """
        class_name = getattr(dcmObj, 'PhotometricInterpretation', None)
        return class_name


trans_manager = PITransManager()
pi_classifier = trans_manager.get_mode
get_pi_trans_func = trans_manager.get_func


def pi_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> [str, np.ndarray]:
    mode, func = trans_manager.get_func(dcmObj)
    if func is not None:
        return mode, func(dcmObj, image_data, depth=depth)
    else:
        return mode, do_nothing(image_data)


@trans_manager.register('MONOCHROME1')
def pi_monochrome1(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    GrayScale: white to black

    Pixel data represent a single monochrome image plane. The minimum sample value is 
    intended to be displayed as white after any VOI gray scale transformations have been
    performed. See PS3.4. This value may be used only when Samples per Pixel (0028,0002)
    has a value of 1. May be used for pixel data in a Native (uncompressed) or Encapsulated
    (compressed) format; see Section 8.2 in PS3.5.

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    return -image_data + depth - 1


@trans_manager.register('MONOCHROME2')
def pi_monochrome2(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    GrayScale: white to black

    Pixel data represent a single monochrome image plane. The minimum sample value is 
    intended to be displayed as black after any VOI gray scale transformations have been
    performed. See PS3.4. This value may be used only when Samples per Pixel (0028,0002)
    has a value of 1. May be used for pixel data in a Native (uncompressed) or Encapsulated
    (compressed) format; see Section 8.2 in PS3.5.

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    return image_data


@trans_manager.register('RGB')
def pi_rgb(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    RGB

    Pixel data represent a color image described by red, green, and blue image planes.
    The minimum sample value for each color plane represents minimum intensity of the color.
    This value may be used only when Samples per Pixel (0028,0002) has a value of 3. Planar
    Configuration (0028,0006) may be 0 or 1. May be used for pixel data in a Native 
    (uncompressed) or Encapsulated (compressed) format; see Section 8.2 in PS3.5.

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    return image_data


def convert_ybr_rgb(image_data: np.ndarray, M: np.ndarray, B: np.ndarray):
    """
    formula: I_ybr = M * I_rgb + B
    for example RGB -> YCBCR

    Y  = + .2990R + .5870G + .1140B
    CB = - .1687R - .3313G + .5000B + 128
    CR = + .5000R - .4187G - .0813B + 128

    M = [
        [ 0.2990,  0.5870,  0.1140],
        [-0.1687, -0.3313,  0.5000],
        [ 0.5000, -0.4187, -0.0813],
    ]
    B = [0,128,128]
    """
    if image_data.shape[-1] != 3:
        raise ValueError(f'image_data should has shape (...,H,W,3), got {image_data.shape}')

    dtype = image_data.dtype
    I_ybr = image_data.astype(np.float64)

    B = np.expand_dims(B, axis=tuple(range(len(I_ybr.shape)-1))) # shape (..., 3)
    M_inv = np.linalg.inv(M)
    I_rgb = np.tensordot(I_ybr-B, M_inv.T, axes=1)
    return I_rgb.clip(0, 255).astype(dtype)


@trans_manager.register('YBR_FULL')
def pi_ybr_full(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    YCbCr 4:4:4

    Pixel data represent a color image described by one luminance (Y) and two chrominance
    planes (CB and CR). This photometric interpretation may be used only when Samples
    per Pixel (0028,0002) has a value of 3. May be used for pixel data in a Native 
    (uncompressed) or Encapsulated (compressed) format; see Section 8.2 in PS3.5 . 
    Planar Configuration (0028,0006) may be 0 or 1.

    This Photometric Interpretation is primarily used with RLE compressed bit streams,
    for which the Planar Configuration (0028,0006) may be 0 or 1; see Section 8.2.2 in
    PS3.5 and Section G.2 in PS3.5 . When used in the US Image Module, the Planar 
    Configuration (0028,0006) is required to be 1; see Section C.8.5.6.1.16 "Planar 
    Configuration".

    Black is represented by Y equal to zero. The absence of color is represented by 
    both CB and CR values equal to half full scale.

    Note
    In the case where Bits Allocated (0028,0100) has value of 8 half full scale is 128.

    In the case where Bits Allocated (0028,0100) has a value of 8 then the following equations
    convert between RGB and YCBCR Photometric Interpretation.

    Y  = + .2990R + .5870G + .1140B
    CB = - .1687R - .3313G + .5000B + 128
    CR = + .5000R - .4187G - .0813B + 128

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    try:
        from pydicom.pixel_data_handlers.util import convert_color_space
    except:
        M = np.array([
            [ 0.2990,  0.5870,  0.1140],
            [-0.1687, -0.3313,  0.5000],
            [ 0.5000, -0.4187, -0.0813],
        ])
        B = np.array([16, 128, 128])
        return convert_ybr_rgb(image_data, M, B)
    else:
        return convert_color_space(image_data, 'YBR_FULL', 'RGB')


@trans_manager.register('YBR_FULL_422')
def pi_ybr_full_422(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    YCbCr 4:2:2

    The same as YBR_FULL except that the CB and CR values are sampled horizontally at half
    the Y rate and as a result there are half as many CB and CR values as Y values.

    Planar Configuration (0028,0006) shall be 0. May be used for pixel data in a Native
    (uncompressed) or Encapsulated (compressed) format; see Section 8.2 in PS3.5 .

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    try:
        from pydicom.pixel_data_handlers.util import convert_color_space
    except:
        M = np.array([
            [ 0.2990,  0.5870,  0.1140],
            [-0.1687, -0.3313,  0.5000],
            [ 0.5000, -0.4187, -0.0813],
        ])
        B = np.array([16, 128, 128])
        return convert_ybr_rgb(image_data, M, B)
    else:
        return convert_color_space(image_data, 'YBR_FULL_422', 'RGB')


@trans_manager.register('YBR_PARTIAL_420', experimental=True)
def pi_ybr_partial_420(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    YCbCr 

    Pixel data represent a color image described by one luminance (Y) and two chrominance
    planes (CB and CR).

    This photometric interpretation may be used only when Samples per Pixel (0028,0002)
    has a value of 3. The CB and CR values are sampled horizontally and vertically at 
    half the Y rate and as a result there are four times less CB and CR values than Y values.

    Planar Configuration (0028,0006) shall be 0. Shall only be used for pixel data in 
    an Encapsulated (compressed) format; see Section 8.2 in PS3.5 .

    Note
    This Photometric Interpretation is primarily used with MPEG compressed bit streams. 
    For a discussion of the sub-sampling notation and siting, see [Poynton 2008].

    Luminance and chrominance values are represented as follows:

    1. black corresponds to Y = 16;
    2. Y is restricted to 220 levels (i.e., the maximum value is 235);
    3. CB and CR each has a minimum value of 16;
    4. CB and CR are restricted to 225 levels (i.e., the maximum value is 240);
    5. lack of color is represented by CB and CR equal to 128.

    In the case where Bits Allocated (0028,0100) has value of 8 then the following 
    equations convert between RGB and YBR_PARTIAL_420 Photometric Interpretation

    Y = + .2568R + .5041G + .0979B + 16
    CB= - .1482R - .2910G + .4392B + 128
    CR= + .4392R - .3678G - .0714B + 128

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    M = np.array([
        [ 0.2568,  0.5041,  0.0979],
        [-0.1482, -0.2910,  0.4392],
        [ 0.4392, -0.3678, -0.0714],
    ])
    B = np.array([16, 128, 128])
    return convert_ybr_rgb(image_data, M, B)


@trans_manager.register('YBR_ICT', experimental=True)
def pi_ybr_ict(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    YCbCr Irreversible Color Transformation

    Pixel data represent a color image described by one luminance (Y) and two chrominance 
    planes (CB and CR).

    This photometric interpretation may be used only when Samples per Pixel (0028,0002)
    has a value of 3. Planar Configuration (0028,0006) shall be 0. Shall only be used 
    for pixel data in an Encapsulated (compressed) format; see Section 8.2 in PS3.5 .

    Note
    This Photometric Interpretation is primarily used with JPEG 2000 compressed bit streams.

    Black is represented by Y equal to zero. The absence of color is represented by both
    CB and CR values equal to zero.

    Regardless of the value of Bits Allocated (0028,0100), the following equations convert
    between RGB and YCBCR Photometric Interpretation.

    Y  = + .29900R + .58700G + .11400B
    CB = - .16875R - .33126G + .50000B
    CR = + .50000R - .41869G - .08131B

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    M = np.array([
        [ 0.29900,  0.58700,  0.11400],
        [-0.16875, -0.33126,  0.50000],
        [ 0.50000, -0.41869, -0.08131],
    ])
    B = np.array([0, 0, 0])
    return convert_ybr_rgb(image_data, M, B)


@trans_manager.register('YBR_RCT', experimental=True)
def pi_ybr_rct(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = 256,
        ) -> np.ndarray:
    """
    YCbCr Reversible Color Transformation

    Pixel data represent a color image described by one luminance (Y) and two chrominance
    planes (CB and CR).

    This photometric interpretation may be used only when Samples per Pixel (0028,0002)
    has a value of 3. Planar Configuration (0028,0006) shall be 0. Shall only be used
    for pixel data in an Encapsulated (compressed) format; see Section 8.2 in PS3.5 .

    Note
    This Photometric Interpretation is primarily used with JPEG 2000 compressed bit streams.

    Black is represented by Y equal to zero. The absence of color is represented by both
    CB and CR values equal to zero.

    Regardless of the value of Bits Allocated (0028,0100), the following equations convert 
    between RGB and YBR_RCT Photometric Interpretation.

    Y = ⌊(R + 2G +B) / 4⌋ (Note: ⌊…⌋ mean floor)
    CB= B - G
    CR= R - G

    The following equations convert between YBR_RCT and RGB Photometric Interpretation.

    G = Y - ⌊ (CR+ CB) / 4⌋
    R = CR+ G
    B = CB+ G

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004

    Argument:
        dcmObj: pydicom.FileDataset
        image_data: (np.ndarray) shape (...,H,W,C)
        depth: (int) 256 for 8-bit 
    Returns:
        image (np.ndarray) keep original shape
    """
    if image_data.shape[-1] != 3:
        raise ValueError(f'image_data should has shape (...,H,W,3), got {image_data.shape}')

    dtype = image_data.dtype
    I_ybr = image_data.astype(np.float64)

    G = I_ybr[...,0] - np.floor((I_ybr[...,1]+I_ybr[...,2])/4)
    I_rgb = np.stack([
        G,
        I_ybr[...,2] + G,
        I_ybr[...,1] + G,
    ], axis=-1)
    return I_rgb.clip(0,256).astype(dtype)
