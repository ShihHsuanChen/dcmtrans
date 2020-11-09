import logging
import pydicom
import numpy as np
import SimpleITK as sitk
from typing import Union


__all__ = ['read_pixel', 'read_pixel_pil', 'read_pixel_pydicom', 'read_pixel_sitk']


def read_pixel(filename: str, return_on_fail=None):
    for method in [read_pixel_pydicom, read_pixel_sitk, read_pixel_pil]:
        try:
            img_arr = method(filename)
            method_name = method.__qualname__
        except Exception as e:
            logging.warning(e)
        else:
            break
    else:
        return return_on_fail
    logging.info(f'Read dicom image by using {method_name}')
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
    return sitk.GetArrayFromImage(itkimg)


def read_pixel_pil(dcm: Union[str, pydicom.dataset.FileDataset]):
    if isinstance(dcm, str):
        data = pydicom.dcmread(dcm, force=True)
    elif isinstance(dcm, pydicom.dataset.FileDataset):
        data = dcm
    else:
        raise TypeError(f'Invalid argument type. Expect str or pydicom.dataset.FileDataset, given {type(dcm)}')
    rows = int(data.Rows)
    cols = int(data.Columns)
    spx = int(data.SamplesPerPixel)
    nbits = int(len(data.PixelData) * 8 / rows / cols / spx)
    slope = data.get('RescaleSlope', None)
    unit = data.get('RescaleType', None)
    modality = data.get('Modality', None)
    modality = str(modality).strip() if modality is not None else None
    signed = slope is not None and int(slope) == 1 and (unit is not None or modality in ['CT', 'MR'])

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
