import os
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np

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
        **kwargs,
        ):
    r"""
    Inputs:
    - volume: (np.ndarray) shape (D,H,W)
    - dilute: (int) Draw image per `dilute` slices. (default: 1)
    - ncols: (int) Maximum number of subplot columns. If D less than ncols, number of 
                   subplot columns will be D. (default: 5)
    - figwidth: (float) figure width (default: 10)
    - metadata: (iterable) text to be shown on each subplots. The length should be equal to D.

    Returns: fig, axs from plt.subplots
    """
    import matplotlib.pyplot as plt

    kwargs = {'cmap': 'gray', **kwargs}
    N = int(np.ceil(volume.shape[0]/dilute))
    ncols = min(ncols, N)
    nrows = int(np.ceil(N/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(figwidth,figwidth/ncols*nrows))
    for i in range(nrows): # nrows
        for j in range(ncols): # ncols
            k = i*ncols+j
            ax = axs.flat[k] if nrows > 1 or ncols > 1 else axs
            if k < N:
                _k = k*dilute
                if metadata is not None:
                    text = metadata[_k] if _k < len(metadata) else ''
                    text = '\n' + text
                else:
                    text = ''
                ax.imshow(volume[_k], **kwargs)
                ax.set_title(f'Index: {_k}{text}')
            ax.axis('off')
    return fig, axs
