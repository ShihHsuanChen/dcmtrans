import pydicom
import numpy

from .trans_method import lut_trans
from .trans_method import window_linear_trans
from .trans_method import window_linear_exact_trans
from .trans_method import window_sigmoid_trans
from .modality_trans import modality_classifier

from .ct_window import get_window as get_ct_window


def voi_classifier(dicom_file: pydicom.FileDataset):
    """
    Get value of interest (voi) transform method name from dicom object

    :param dicom_file: object read from pydicom
    :return: voi transform method name. Includes
            'LINEAR', 'LINEAR_EXACT', 'SIGMOID', 'TABLE', or None
    """
    VOILUTFunction = dicom_file.get('VOILUTFunction')
    VOILUTSequence = dicom_file.get('VOILUTSequence')
    WindowCenter = dicom_file.get('WindowCenter')

    if (VOILUTSequence is not None) and (WindowCenter is None):
        return 'TABLE'
    elif WindowCenter is not None:
        if VOILUTFunction is None:
            return 'LINEAR'
        else:
            return VOILUTFunction
    else:
        return None


def _get_window(dicom_file: pydicom.FileDataset, window='default', unit=None):
    if isinstance(window, dict):
        wc = window.get('window_center')
        ww = window.get('window_width')
    elif window == 'default' or unit is None:
        wc = dicom_file.get('WindowCenter')
        ww = dicom_file.get('WindowWidth')
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = wc[0]
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = ww[0]
    elif isinstance(window, str) and unit == 'HU':
        wc, ww = get_ct_window(window)
    else:
        try:
            wc = dicom_file.get('WindowCenter')
            ww = dicom_file.get('WindowWidth')
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


def voi_linear_trans(
        dicom_file: pydicom.FileDataset,
        image_data: numpy.ndarray,
        depth: int,
        window='default',
        unit: str=None
        ):
    window_center, window_width = _get_window(dicom_file, window=window, unit=unit)
    return window_linear_trans(image_data, window_center, window_width, depth)


def voi_linear_exact_trans(
        dicom_file: pydicom.FileDataset, 
        image_data: numpy.ndarray,
        depth: int,
        window='default',
        unit: str=None
        ):
    window_center, window_width = _get_window(dicom_file, window, unit=unit)
    return window_linear_exact_trans(image_data, window_center, window_width, depth)


def voi_sigmoid_trans(
        dicom_file: pydicom.FileDataset,
        image_data: numpy.ndarray,
        depth,
        window='default',
        unit: str=None
        ):
    window_center, window_width = _get_window(dicom_file, window, unit=unit)
    return window_sigmoid_trans(image_data, window_center, window_width, depth)


def voi_lut_trans(
        dicom_file: pydicom.FileDataset,
        image_data: numpy.ndarray, 
        depth,
        window='default',
        unit: str=None
        ):
    VOILUTSequence = getattr(dicom_file, 'VOILUTSequence', [{}])[0]
    lut_descriptor = VOILUTSequence.get('LUTDescriptor')

    if isinstance(lut_descriptor, bytes):
        PixelRepresentation = int(dicom_file.get('PixelRepresentation'))
        if modality_classifier(dicom_file) is None and PixelRepresentation == 0:
            dtype = numpy.ushort
        else:
            dtype = numpy.short
        lut_descriptor = numpy.frombuffer(lut_descriptor, dtype)

    lut_data = VOILUTSequence.get('LUTData')
    if isinstance(lut_data, bytes):
        if lut_descriptor[2] == 8:
            dtype = numpy.uint8
        elif lut_descriptor[2] == 16:
            dtype = numpy.uint16
        else:
            raise ValueError(f'LUTDescriptor[2] should be 8 or 16. Got {lut_descriptor[2]}')
        lut_data = numpy.frombuffer(lut_data, dtype)
    scale_factor = depth / int(2**getattr(dicom_file, 'BitsStored', 12))
    return lut_trans(image_data, lut_descriptor, lut_data, scale_factor)
