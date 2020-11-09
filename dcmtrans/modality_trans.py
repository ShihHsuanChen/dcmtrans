import numpy
import pydicom

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


def modality_linear_trans(dicom_file: pydicom.FileDataset, image_data: numpy.ndarray):
    intercept = getattr(dicom_file, 'RescaleIntercept', None)
    slope = getattr(dicom_file, 'RescaleSlope', None)
    unit = getattr(dicom_file, 'RescaleType', None)
    if unit is None and getattr(dicom_file, 'Modality', '').strip() == 'CT':
        unit = 'HU'

    return linear_trans(image_data, intercept, slope), unit


def modality_lut_trans(dicom_file: pydicom.FileDataset, image_data: numpy.ndarray):
    ModalityLUTSequence = dicom_file.get('ModalityLUTSequence')[0]
    lut_descriptor = ModalityLUTSequence.get('LUTDescriptor')
    if lut_descriptor is None: # debug
        lut_descriptor = ModalityLUTSequence.get('LUTDescriptor') # if don't do this line, line 40 will be None

    if isinstance(lut_descriptor, bytes):
        PixelRepresentation = int(dicom_file.get('PixelRepresentation'))
        dtype = numpy.ushort if PixelRepresentation == 0 else numpy.short
        lut_descriptor = numpy.frombuffer(lut_descriptor, dtype)

    lut_data = ModalityLUTSequence.get('LUTData')
    if isinstance(lut_data, bytes):
        if lut_descriptor[2] == 8:
            dtype = numpy.uint8
        elif lut_descriptor[2] == 16:
            dtype = numpy.uint16
        else:
            raise ValueError(f'LUTDescriptor[2] should be 8 or 16. Got {lut_descriptor[2]}')
        lut_data = numpy.frombuffer(lut_data, dtype)
    unit = ModalityLUTSequence.get('ModalityLUTType')
    return lut_trans(image_data, lut_descriptor, lut_data), unit
