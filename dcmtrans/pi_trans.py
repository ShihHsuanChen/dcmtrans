import numpy as np
import pydicom

from .base import BaseTransManager
from .trans_method import do_nothing


__all__ = [
    'pi_trans',
    'pi_classifier',
    'pi_monochrome1',
    'pi_monochrome2',
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
        depth: int,
        ) -> [str, np.ndarray]:
    mode, func = trans_manager.get_func(dcmObj)
    if func is not None:
        return mode, func(dcmObj, image_data, depth)
    else:
        return mode, do_nothing(image_data)


@trans_manager.register('MONOCHROME1')
def pi_monochrome1(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        nbits: int,
        ) -> np.ndarray:
    """
    GrayScale: white to black

    Pixel data represent a single monochrome image plane. The minimum sample value is 
    intended to be displayed as white after any VOI gray scale transformations have been
    performed. See PS3.4. This value may be used only when Samples per Pixel (0028,0002)
    has a value of 1. May be used for pixel data in a Native (uncompressed) or Encapsulated
    (compressed) format; see Section 8.2 in PS3.5.

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004
    """
    return -image_data + nbits - 1


@trans_manager.register('MONOCHROME2')
def pi_monochrome2(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        nbits: int,
        ) -> np.ndarray:
    """
    GrayScale: white to black

    Pixel data represent a single monochrome image plane. The minimum sample value is 
    intended to be displayed as black after any VOI gray scale transformations have been
    performed. See PS3.4. This value may be used only when Samples per Pixel (0028,0002)
    has a value of 1. May be used for pixel data in a Native (uncompressed) or Encapsulated
    (compressed) format; see Section 8.2 in PS3.5.

    Reference: https://dicom.innolitics.com/ciods/segmentation/image-pixel/00280004
    """
    return image_data
