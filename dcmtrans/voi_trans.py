import pydicom
import numpy as np
from typing import Optional

from .base import BaseTransManager
from .trans_method import do_nothing
from .trans_method import lut_trans
from .trans_method import window_linear_trans
from .trans_method import window_linear_exact_trans
from .trans_method import window_sigmoid_trans
from .modality_trans import trans_manager as modality_trans_manager

from .ct_window import get_window as get_ct_window


__all__ = [
    'voi_trans',
    'voi_classifier',
    'voi_linear_trans',
    'voi_linear_exact_trans',
    'voi_sigmoid_trans',
    'voi_lut_trans',
    'get_voi_trans_func',
]


class VOITransManager(BaseTransManager):
    def get_mode(self, dcmObj: pydicom.FileDataset) -> str:
        """
        Get value of interest (voi) transform method name from dicom object

        :param dcmObj: object read from pydicom
        :return: voi transform method name. Includes
                'LINEAR', 'LINEAR_EXACT', 'SIGMOID', 'TABLE', or None
        """
        VOILUTFunction = dcmObj.get('VOILUTFunction')
        VOILUTSequence = dcmObj.get('VOILUTSequence')
        WindowCenter = dcmObj.get('WindowCenter')

        if (VOILUTSequence is not None) and (WindowCenter is None):
            return 'TABLE'
        elif WindowCenter is not None:
            if VOILUTFunction is None:
                return 'LINEAR'
            else:
                return VOILUTFunction
        else:
            return 'LINEAR'


trans_manager = VOITransManager()
voi_classifier = trans_manager.get_mode
get_voi_trans_func = trans_manager.get_func


def voi_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int,
        unit: Optional[str] = None,
        window = 'default',
        ) -> [str, np.ndarray]:
    mode, func = trans_manager.get_func(dcmObj)
    if func is not None:
        return mode, func(dcmObj, image_data, depth, window=window, unit=unit)
    else:
        return mode, do_nothing(image_data)


def _get_window(dcmObj: pydicom.FileDataset, window='default', unit=None):
    if isinstance(window, dict):
        wc = window.get('window_center')
        ww = window.get('window_width')
    elif window == 'default' or unit is None:
        try:
            wc = dcmObj.WindowCenter
            ww = dcmObj.WindowWidth
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]
        except:
            BitsStored = dcmObj.get('BitsStored')
            wc = 2 ** (BitsStored - 1)
            ww = 2 ** BitsStored
    elif isinstance(window, str) and unit in ['HU', 'Hounsfield Unit']:
        wc, ww = get_ct_window(window)
    else:
        try:
            wc = dcmObj.WindowCenter
            ww = dcmObj.WindowWidth
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]
        except:
            wc = None
            ww = None

    if wc is None or ww is None:
        raise ValueError('Either window center or window width not found')
    return wc, ww


@trans_manager.register('LINEAR')
def voi_linear_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int,
        window='default',
        unit: Optional[str] = None,
        ) -> np.ndarray:
    window_center, window_width = _get_window(dcmObj, window=window, unit=unit)
    return window_linear_trans(image_data, window_center, window_width, depth)


@trans_manager.register('LINEAR_EXACT')
def voi_linear_exact_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int,
        window='default',
        unit: Optional[str] = None,
        ) -> np.ndarray:
    window_center, window_width = _get_window(dcmObj, window, unit=unit)
    return window_linear_exact_trans(image_data, window_center, window_width, depth)


@trans_manager.register('SIGMOID')
def voi_sigmoid_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth,
        window='default',
        unit: Optional[str] = None,
        ) -> np.ndarray:
    window_center, window_width = _get_window(dcmObj, window, unit=unit)
    return window_sigmoid_trans(image_data, window_center, window_width, depth)


@trans_manager.register('TABLE')
def voi_lut_trans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth,
        window='default',
        unit: Optional[str] = None,
        ) -> np.ndarray:
    VOILUTSequence = getattr(dcmObj, 'VOILUTSequence', [{}])[0]
    lut_descriptor = VOILUTSequence.get('LUTDescriptor')
    if lut_descriptor is None:
        lut_descriptor = VOILUTSequence.get('LUTDescriptor')

    if isinstance(lut_descriptor, bytes):
        PixelRepresentation = int(dcmObj.get('PixelRepresentation'))
        if modality_trans_manager.get_mode(dcmObj) is None and PixelRepresentation == 0:
            dtype = np.ushort
        else:
            dtype = np.short
        lut_descriptor = np.frombuffer(lut_descriptor, dtype)

    lut_data = VOILUTSequence.get('LUTData')
    if isinstance(lut_data, bytes):
        if lut_descriptor[2] == 8:
            dtype = np.uint8
        elif lut_descriptor[2] == 16:
            dtype = np.uint16
        else:
            raise ValueError(f'LUTDescriptor[2] should be 8 or 16. Got {lut_descriptor[2]}')
        lut_data = np.frombuffer(lut_data, dtype)
    scale_factor = depth / int(2**getattr(dcmObj, 'BitsStored', 12))
    return lut_trans(image_data, lut_descriptor, lut_data, scale_factor)
