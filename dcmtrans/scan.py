import os
import logging
import traceback
from pathlib import Path
from types import GeneratorType
from typing import Dict, Any, Iterable, Generator, Union
from collections import namedtuple
import pydicom

from .typing import PathLike
from .reconstruction import reconstruct_series, RecInfo


InstanceKey = namedtuple('InstanceKey', ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID'])


_logger = logging.getLogger(__name__)


def collect_dicoms(
        scanner: Union[Generator, Iterable],
        verbose: bool = False,
        ) -> Dict[InstanceKey, Dict[str, pydicom.FileDataset]]:
    collections = dict() # {<InstanceKey object>: {<path>: <dcmobj>}}}
    for p in scanner:
        if not os.path.isfile(p):
            continue
        try:
            dcmobj = pydicom.dcmread(p)
        except Exception as e:
            if verbose:
                _logger.debug(f'Cannot read file {p}')
                _logger.debug(str(e))
                _logger.debug(traceback.format_exc())
            continue
        if not hasattr(dcmobj, 'SeriesInstanceUID'):
            if verbose:
                _logger.debug(f'Cannot find attribute "SeriesInstanceUID" from file {p}')
            continue
        key = InstanceKey(*[str(getattr(dcmobj, tag, '')) for tag in InstanceKey._fields])
        suid = str(dcmobj.SeriesInstanceUID)
        if key not in collections:
            collections[key] = dict()
        collections[key][p] = dcmobj
    return collections


def collect_dicoms_reconstuct(
        scanner: Union[Generator, Iterable],
        verbose: bool = False,
        **kwargs,
        ) -> Dict[InstanceKey, RecInfo]:
    collections = collect_dicoms(scanner, verbose=verbose)
    # reconstruct series
    series_dict = dict()
    for key, dcm_dict in collections.items():
        try:
            rec_info = reconstruct_series(dcm_dict, **kwargs)
        except Exception as e:
            if verbose:
                _logger.warn(str(e))
                _logger.warn(traceback.format_exc())
        else:
            series_dict[key] = rec_info
    return series_dict


def _dir_scanner(topdir):
    for root, ds, fs in os.walk(topdir):
        for fname in fs:
            yield os.path.join(root, fname)


def scan_directory(topdir: str) -> Dict[InstanceKey, Dict[str, pydicom.FileDataset]]:
    return collect_dicoms(_dir_scanner(topdir))


def scan_directory_reconstuct(topdir: str, **kwargs) -> Dict[InstanceKey, RecInfo]:
    return collect_dicoms_reconstuct(_dir_scanner(topdir), **kwargs)
