import os
import pytest
import numpy as np

from dcmtrans import dcm_imread


file_normal = 'D:\\dataset\\dataset_kaggle_RSNA_Pneumonia_Detection_train\\1_nodule\\00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dcm'
file_report = 'D:\\dataset\\_issue_DICOM\\_issue_DICOM\\train\\normal\\report04.dcm'
file_path_twzh = u'D:\\dataset\\_issue_DICOM1\\中文\\train\\normal\\image09.dcm'
file_dir_twzh = 'D:\\dataset\\_issue_DICOM1'


def test_read_pixel():
    assert isinstance(dcm_imread.read_pixel(file_normal, None), np.ndarray)
    assert dcm_imread.read_pixel(file_report, None) is None


def test_read_pixel_pydicom():
    assert isinstance(dcm_imread.read_pixel_pydicom(file_normal), np.ndarray)
    assert isinstance(dcm_imread.read_pixel_pydicom(file_path_twzh), np.ndarray)
    # assert isinstance(dcm_imread.read_pixel_pydicom(file_path_twzh), np.array)


def test_read_pixel_sitk():
    print(file_path_twzh, file_dir_twzh)
    for root, dirs, files in os.walk(file_dir_twzh):
        if 'image09.dcm' in files:
            path = os.path.join(root, 'image09.dcm')
            print(path)
            img = dcm_imread.read_pixel_sitk(path)
            assert isinstance(img, np.ndarray)
            raise ValueError(str(img))

    assert isinstance(dcm_imread.read_pixel_sitk(file_normal), np.ndarray)
    assert isinstance(dcm_imread.read_pixel_sitk(file_path_twzh), np.ndarray)
    # assert isinstance(dcm_imread.read_pixel_sitk(file_path_twzh), np.array)


def test_read_pixel_bytes():
    # assert isinstance(dcm_imread.read_pixel_bytes(file_normal), np.ndarray)
    # assert isinstance(dcm_imread.read_pixel_bytes(file_path_twzh), np.array)
    pass
