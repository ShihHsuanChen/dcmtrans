import pydicom
import traceback
import PIL.Image
import numpy as np

from typing import Union, Dict, List, Any, Optional

from .voi_trans import voi_trans
from .modality_trans import modality_trans
from .pi_trans import pi_trans

from .reconstruction import reconstruct_series
from .reconstruction import get_instance_info

from .formats import get_nbits_from_colormap


DEFULT_COLORMAP = 'L'
BITS_TRANS = get_nbits_from_colormap(DEFULT_COLORMAP)
DEPTH = int(2**BITS_TRANS)


WindowType = Union[
    str, # examples: 'lung', 'abdomen' .etc. see dcmtrans.CT_PRESET_WIINDOW
    Dict[str, Union[int, float]], # {'window_center': window center, 'window_width': window width}
]

def dcmtrans(
        dcmObj: pydicom.FileDataset,
        image_data: np.ndarray,
        depth: int = DEPTH,
        window: Optional[List[WindowType]] = ('default',)
        ) -> [List[Union[None, np.ndarray]], List[Union[None, Exception]], Dict[str, str]]:
    """
    :param dcmObj: object read from pydicom
    :param image_data: image array
    :param depth: output image bit-depth. example: 256 (2**8)
    :param window: given windows. [<window str, dict, None>]
        <window None>: don't apply any window
        <window str>: examples: 'lung', 'abdomen' .etc. see dcmtrans.CT_PRESET_WIINDOW
        <window dict>: {'window_center': window center, 'window_width': window width}
                       for example: {'window_center': -750, 'window_width': 700}
    :return: [image_data | None], [exception | None], {'modality': mod_mode, 'modality_unit': unit, 'voi': voi_mode, 'pi': pi_mode}
    """
    mod_mode, t_image_data, unit = modality_trans(dcmObj, image_data)

    image_data_list = list()
    exception_list = list()
    voi_mode = None
    pi_mode = None

    if window is None:
        image_data_list = [t_image_data]
    else:
        for w in window:
            try:
                m_image_data = np.copy(t_image_data)
                voi_mode, m_image_data = voi_trans(dcmObj, m_image_data, depth, unit=unit, window=w)
                pi_mode, m_image_data = pi_trans(dcmObj, m_image_data, depth)
                exc = None
            except Exception as e:
                m_image_data = None
                exc = e
            exception_list.append(exc)
            image_data_list.append(m_image_data)

    return image_data_list, exception_list, {'modality': mod_mode, 'modality_unit': unit, 'voi': voi_mode, 'pi': pi_mode}


def image_convert(image_arr: np.ndarray, dtype=np.uint8, fmt: Union[str, None] = None, to_fmt: Union[str, None] = None) -> PIL.Image.Image:
    """
    dimemsion:
    L:     (# pixel y, # pixel x)
    RGB:   ([R,G,B], # pixel y, # pixel x)
    HSV:   ([H,S,V], # pixel y, # pixel x)
    YCbCr: ([Y,Cb,Cr], # pixel y, # pixel x)
    RGBA:  ([R,G,B], # pixel y, # pixel x)
    CMYK:  ([C,M,Y,K], # pixel y, # pixel x)

    :param image_arr: image data. example: image_arr.shape = (1, 640, 480)
    :param fmt: input image format code
    :param to_fmt: output image format code
    :return: PIL.Image object
    """
    arr = np.asarray(image_arr, dtype=dtype)
    dim = arr.shape
    if len(dim) == 2:
        _fmt = fmt or 'L'
    elif len(dim) == 3:
        if dim[0] == 1:
            arr = arr[0, :, :]
            _fmt = fmt or 'L'
        elif dim[0] == 3:
            arr = arr.transpose((1, 2, 0))
            _fmt = fmt or 'RGB'
        elif dim[0] == 4:
            arr = arr.transpose((1, 2, 0))
            _fmt = fmt or 'CMYK'
        else:
            raise ValueError('image_arr has shape[0] > 4')
    else:
        raise ValueError('image_arr has len(shape) > 3')
    image = PIL.Image.fromarray(arr, _fmt)
    if to_fmt is not None:
        image = image.convert(to_fmt)
    return image
