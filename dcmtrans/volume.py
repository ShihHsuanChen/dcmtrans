import os
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List, Tuple, Union

import numpy as np

from .typing import PathLike
from .dcmtrans import dcmtrans, DEPTH, WindowType
from .dcm_imread import read_pixel
from .reconstruction import RecInfo


__all__ = ['build_volume_from_recon_info', 'plot_volume']


def build_volume_from_recon_info(
        rec_info: RecInfo[PathLike],
        resize: Union[int, Tuple[int, int], None] = None,
        resize_interpolation: int = cv2.INTER_LINEAR,
        depth: int = DEPTH,
        window: Optional[WindowType] = None, # TODO multiple window?
        ) -> np.ndarray:
    r"""
    Inputs:
    - rec_info: RecInfo[PathLike] output from dcmtrans.reconstruction.reconstruct_series
        Notice that the value of 
    - resize: (tuple, optinal) resize to (H,W), if None, not resize
    - depth: (int) output image bit-depth. example: 256 (2**8)
    - window: given SINGEL window. [<window str, dict>]
        <window str>: examples: 'lung', 'abdomen' .etc. see dcmtrans.CT_PRESET_WIINDOW
        <window dict>: {'window_center': window center, 'window_width': window width}
                       for example: {'window_center': -750, 'window_width': 700}

    Return:
    - volume: (np.ndarray) Shape (D,*,H,W)
    """
    if resize is not None:
        if isinstance(resize, int):
            resize = (resize, resize)
        if len(resize) != 2:
            raise ValueError(f'Invalid resize format {resize}')
        _resize = (resize[1], resize[0])
    if window is not None:
        window = [window] # TODO: multiple window?

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
        arrs, e, _info = dcmtrans(dcmobj, arr, depth=depth, window=window)
        # resize
        if resize is not None:
            arrs = [cv2.resize(arr, _resize, interpolation=resize_interpolation) for arr in arrs]
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
