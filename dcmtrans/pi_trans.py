import numpy
import pydicom


def pi_classifier(dicom_file: pydicom.FileDataset) -> str:
    """
    Get photometric interpretation (pi) transform method name from dicom object

    :param dicom_file: object read from pydicom
    :return: pi transform method name. return value can be one of follows:
             MONOCHROME1, MONOCHROME2, PALETTE COLOR, RGB, ...
             Details can be found here https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004.
             However this module can only deal with class of MONOCHROME1, and MONOCHROME2
    """
    class_name = getattr(dicom_file, 'PhotometricInterpretation', None)
    return class_name


def pi_monochrome1(
        dicom_file: pydicom.FileDataset,
        image_data: numpy.ndarray,
        nbits: int
        ) -> numpy.ndarray:
    new_image_data = (-image_data + nbits - 1)
    return new_image_data


def pi_monochrome2(
        dicom_file: pydicom.FileDataset,
        image_data: numpy.ndarray,
        nbits: int
        ) -> numpy.ndarray:
    return image_data
