import pydicom
import numpy as np
from typing import Tuple, Optional

from .trans_method import lut_trans, linear_trans


def modality_classifier(dicom_file):
    """
    Get modality transform method name from dicom object

    :param dicom_file: object read from pydicom
    :return: modality transform method name. Includes
            'LINEAR', 'TABLE', or None
    """
    ModalityLUTSequence = getattr(dicom_file, 'ModalityLUTSequence', None)
    RescaleIntercept = getattr(dicom_file, 'RescaleIntercept', None)

    if (ModalityLUTSequence is not None) and (RescaleIntercept is None):
        return 'TABLE'
    elif RescaleIntercept is not None:
        return 'LINEAR'
    else:
        return None


def modality_linear_trans(
        dicom_file: pydicom.FileDataset,
        image_data: np.ndarray,
        ) -> Tuple[np.ndarray, Optional[str]]:
    intercept = getattr(dicom_file, 'RescaleIntercept', None)
    slope = getattr(dicom_file, 'RescaleSlope', None)
    unit = getattr(dicom_file, 'RescaleType', None)
    if unit is None and getattr(dicom_file, 'Modality', '').strip() == 'CT':
        unit = 'HU'
    return linear_trans(image_data, intercept, slope), unit


def modality_lut_trans(
        dicom_file: pydicom.FileDataset,
        image_data: np.ndarray,
        ) -> Tuple[np.ndarray, Optional[str]]:
    ModalityLUTSequence = dicom_file.get('ModalityLUTSequence')[0]
    lut_descriptor = ModalityLUTSequence.get('LUTDescriptor')
    if lut_descriptor is None: # debug
        lut_descriptor = ModalityLUTSequence.get('LUTDescriptor') # if don't do this line, line 40 will be None

    if isinstance(lut_descriptor, bytes):
        PixelRepresentation = int(dicom_file.get('PixelRepresentation'))
        dtype = np.ushort if PixelRepresentation == 0 else np.short
        lut_descriptor = np.frombuffer(lut_descriptor, dtype)

    lut_data = ModalityLUTSequence.get('LUTData')
    if isinstance(lut_data, bytes):
        if lut_descriptor[2] == 8:
            dtype = np.uint8
        elif lut_descriptor[2] == 16:
            dtype = np.uint16
        else:
            raise ValueError(f'LUTDescriptor[2] should be 8 or 16. Got {lut_descriptor[2]}')
        lut_data = np.frombuffer(lut_data, dtype)
    unit = ModalityLUTSequence.get('ModalityLUTType')
    return lut_trans(image_data, lut_descriptor, lut_data), unit
