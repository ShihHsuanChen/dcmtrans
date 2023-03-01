import os
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt

from .typing import PathLike
from .dcmtrans import dcmtrans
from .dcm_imread import read_pixel
from .reconstruction import RecInfo


__all__ = ['build_volume_from_recon_info', 'plot_volume']


def build_volume_from_recon_info(rec_info: RecInfo[PathLike]) -> np.ndarray:
    r"""
    Inputs:
    - rec_info: RecInfo[PathLike] output from dcmtrans.reconstruction.reconstruct_series
        Notice that the value of 

    Return:
    - volume: (np.ndarray) Shape (D,*,H,W)
    """
    # read image, transform
    img_list = []
    for idx in rec_info.index_list:
        path = rec_info.index_map[idx]
        # convert path to string
        if isinstance(path, Path):
            path = str(path)
        # read from pydicom.FileDataset if value of index_map is not a path
        # (should not happen. just in case)
        dcmobj = rec_info.index_dicom_dict[idx]
        if os.path.isfile(path):
            arr = read_pixel(path)
        else:
            arr = dcmobj.pixel_array
        arrs, e, _info = dcmtrans(dcmobj, arr, window=None)
        img_list.append(arrs[0])
    volume = np.stack(img_list)
    return volume


def plot_volume(
        volume: np.ndarray,
        dilute: int = 1,
        ncols: int = 5,
        figwidth: float = 10,
        metadata: Optional[Iterable[Any]] = None,
        ):
    N = volume.shape[0] // dilute
    nrows = int(np.ceil(N/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(figwidth,figwidth/ncols*nrows))
    for i, _axs in enumerate(axs): # nrows
        for j, ax in enumerate(_axs): # ncols
            k = i*ncols+j
            if k < N:
                _k = k*dilute
                if metadata is not None:
                    text = metadata[_k] if _k < len(metadata) else ''
                    text = '\n' + text
                else:
                    text = ''
                ax.imshow(volume[_k], cmap='gray')
                ax.set_title(f'Index: {_k}{text}')
            ax.axis('off')
    return fig, axs
