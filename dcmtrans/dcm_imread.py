import PIL
import logging
import pydicom
import numpy as np
import SimpleITK as sitk
from typing import Union


__all__ = ['read_pixel', 'read_pixel_bytes', 'read_pixel_pydicom', 'read_pixel_sitk']


logger = logging.getLogger(__name__)


def read_pixel(filename: str, return_on_fail=None):
    for method in [read_pixel_pydicom, read_pixel_sitk, read_pixel_bytes]:
        try:
            img_arr = method(filename)
            method_name = method.__qualname__
        except Exception as e:
            logger.warning(e)
        else:
            break
    else:
        return return_on_fail
    logger.debug(f'Read dicom image by using {method_name}')
    return img_arr


def read_pixel_pydicom(dcm: Union[str, pydicom.dataset.FileDataset]):
    if isinstance(dcm, str):
        data = pydicom.dcmread(dcm, force=True)
    elif isinstance(dcm, pydicom.dataset.FileDataset):
        data = dcm
    else:
        raise TypeError(f'Invalid argument type. Expect str or pydicom.dataset.FileDataset, given {type(dcm)}')
    return data.pixel_array


def read_pixel_sitk(filename: str):
    itkimg = sitk.ReadImage(filename)
    try:
        RescaleSlope = itkimg.GetMetaData('0028|1053')
        RescaleSlope = float(str(RescaleSlope).strip())
    except Exception as e:
        logger.warning(e)
        RescaleSlope = 1
    try:
        RescaleIntercept = itkimg.GetMetaData('0028|1052')
        RescaleIntercept = float(str(RescaleIntercept).strip())
    except Exception as e:
        logger.warning(e)
        RescaleIntercept = 0
    arr = sitk.GetArrayFromImage(itkimg)
    arr = (arr - float(RescaleIntercept)) / float(RescaleSlope)
    # sitk outputs shape is (?,H,W) while pydicom gives (H,W) for CT images
    # TODO need to test for other modalities
    return arr.squeeze()


def read_pixel_bytes(dcm: Union[str, pydicom.dataset.FileDataset]):
    if isinstance(dcm, str):
        data = pydicom.dcmread(dcm, force=True)
    elif isinstance(dcm, pydicom.dataset.FileDataset):
        data = dcm
    else:
        raise TypeError(f'Invalid argument type. Expect str or pydicom.dataset.FileDataset, given {type(dcm)}')
    rows = int(data.Rows)
    cols = int(data.Columns)
    spx = int(data.SamplesPerPixel)
    nbits = len(data.PixelData) * 8 / rows / cols / spx
    slope = data.get('RescaleSlope', None)
    unit = data.get('RescaleType', None)
    modality = data.get('Modality', None)
    modality = str(modality).strip() if modality is not None else None
    signed = slope is not None and int(slope) == 1 and (unit is not None or modality in ['CT', 'MR'])

    if not nbits.is_integer():
        if spx == 1:
            fmt = 'L'
        elif spx == 3:
            fmt = 'RGB'
        elif spx == 4:
            fmt = 'CMYK'
        else:
            raise ValueError('SamplesPerPixel is larger than 4')
        img = PIL.Image.frombuffer(fmt, (rows, cols), data.PixelData, 'raw')
        img_arr = np.array(img)
    else:
        nbits = int(nbits)
        if nbits == 8 and not signed:
            dtype = np.uint8
        elif nbits == 8 and signed:
            dtype = np.int8
        elif nbits == 16 and not signed:
            dtype = np.uint16
        elif nbits == 16 and signed:
            dtype = np.int16
        else:
            dtype = getattr(np, f'uint{nbits}')
        img_arr = np.frombuffer(data.PixelData, dtype)
        img_arr = img_arr.reshape([rows, cols])
    return img_arr
