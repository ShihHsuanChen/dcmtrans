import pydicom
import numpy as np
from typing import Tuple, Optional

from .base import BaseTransManager
from .trans_method import lut_trans, linear_trans, do_nothing


__all__ = [
    'modality_trans',
    'modality_classifier',
    'modality_linear_trans',
    'modality_lut_trans',
    'get_modality_trans_func',
]


class ModalityTransManager(BaseTransManager):
    def get_mode(self, dcmObj: pydicom.FileDataset) -> str:
        """
        Get modality transform method name from dicom object

        :param dcmObj: object read from pydicom
        :return: modality transform method name. Includes
                'LINEAR', 'TABLE', or None
        """
        ModalityLUTSequence = getattr(dcmObj, 'ModalityLUTSequence', None)
        RescaleIntercept = getattr(dcmObj, 'RescaleIntercept', None)

        if (ModalityLUTSequence is not None) and (RescaleIntercept is None):
            return 'TABLE'
        elif RescaleIntercept is not None:
            return 'LINEAR'
        else:
            return None


trans_manager = ModalityTransManager()
modality_classifier = trans_manager.get_mode
get_modality_trans_func = trans_manager.get_func


def modality_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        ) -> [str, np.ndarray, str]:
    mode, func = trans_manager.get_func(dcmObj)
    if func is not None:
        image_data, unit = func(dcmObj, image_data)
        return mode, image_data, unit
    else:
        return mode, do_nothing(image_data), None


@trans_manager.register('LINEAR')
def modality_linear_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        ) -> Tuple[np.ndarray, Optional[str]]:
    intercept = getattr(dcmObj, 'RescaleIntercept', None)
    slope = getattr(dcmObj, 'RescaleSlope', None)
    unit = getattr(dcmObj, 'RescaleType', None)
    if unit is None and getattr(dcmObj, 'Modality', '').strip() == 'CT':
        unit = 'HU'
    return linear_trans(image_data, intercept, slope), unit


@trans_manager.register('TABLE')
def modality_lut_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        ) -> Tuple[np.ndarray, Optional[str]]:
    ModalityLUTSequence = dcmObj.get('ModalityLUTSequence')[0]
    lut_descriptor = ModalityLUTSequence.get('LUTDescriptor')
    if lut_descriptor is None: # debug
        lut_descriptor = ModalityLUTSequence.get('LUTDescriptor') # if don't do this line, line 40 will be None

    if isinstance(lut_descriptor, bytes):
        PixelRepresentation = int(dcmObj.get('PixelRepresentation'))
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
