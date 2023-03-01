import os
import  traceback
from typing import Dict, Any
from collections import namedtuple
import pydicom

from .reconstruction import reconstruct_series, RecInfo


InstanceKey = namedtuple('InstanceKey', ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID'])


def scan_directory(topdir: str) -> Dict[InstanceKey, Dict[str, pydicom.FileDataset]]:
    collections = dict() # {<InstanceKey object>: {<path>: <dcmobj>}}}
    # read dicom files 
    for root, ds, fs in os.walk(topdir):
        for fname in fs:
            p = os.path.join(root, fname)
            try:
                dcmobj = pydicom.dcmread(p)
            except Exception as e:
                print(f'Cannot read file {p}. {e}')
                print(tracebace.format_exc())
                continue
            if not hasattr(dcmobj, 'SeriesInstanceUID'):
                print(f'Cannot find attribute "SeriesInstanceUID" from file {p}')
                continue
            key = InstanceKey(*[str(getattr(dcmobj, tag, '')) for tag in InstanceKey._fields])
            suid = str(dcmobj.SeriesInstanceUID)
            if key not in collections:
                collections[key] = dict()
            collections[key][p] = dcmobj
    return collections


def scan_directory_reconstuct(topdir: str) -> Dict[InstanceKey, RecInfo]:
    collections = scan_directory(topdir)
    # reconstruct series
    series_dict = dict()
    for key, dcm_dict in collections.items():
        try:
            rec_info = reconstruct_series(dcm_dict)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        else:
            series_dict[key] = rec_info
    return series_dict
