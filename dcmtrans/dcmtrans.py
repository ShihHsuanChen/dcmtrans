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


# TODO
def reconstruct(tank: List[Dict[str, Any]]) -> Dict:
    """
    :param tank:
        [{
            series_no: <series_no>
            instance_no: <instance_no>
            dcmObj: <dicom object (FileDataset)>
            image_arr: <image array (numpy.ndarray) not important>
            fname: <file name>
        }]
    :return: result
        {
            <series_no>: {
                info: {
                    series_no:
                    description:
                    N_pixel_i:
                    N_pixel_j:
                    N_slice:
                    slice_thickness:
                    spacing_i:
                    spacing_j:
                    spacing_slice:
                    orient:
                    orient_like:
                    patient_position:
                    use_contrast:
                    chirality:
                    }
                objs: {
                    <instance_no>: {
                        fname: <file name>
                        dcmObj: <dicom object (FileDataset)>
                        image_arr: <image array (numpy.ndarray) not important>
                        info: {
                            bits_origin:
                            image_position:
                        }
                    }
                }
            }
        }
    """
    result = dict()
    series_no_list = {x.get('series_no') for x in tank}
    modality = list()

    for series_no in series_no_list:
        result[series_no] = {'info': dict(), 'objs': dict()}

        inst_list = [x for x in tank if x.get('series_no') == series_no]
        inst_dict = {x.get('instance_no'): x.get('dcmObj') for x in inst_list}

        try:
            rec_data = reconstruct_series(inst_dict)
            rec_data = rec_data.dict()
        except AssertionError as e:
            obj = inst_dict.get(list(inst_dict.keys())[0])
            result[series_no]['info'] = {
                    'series_no': series_no,
                    'description': getattr(obj, 'SeriesDescription', 'DefaultForNone'),
                    'N_slice': len(inst_dict),
                    'ERROR': list(e.args)
                    }
            continue
        except Exception as e:
            obj = inst_dict.get(list(inst_dict.keys())[0])
            result[series_no]['info'] = {
                    'series_no': series_no,
                    'description': getattr(obj, 'SeriesDescription', 'DefaultForNone'),
                    'N_slice': len(inst_dict),
                    'ERROR': traceback.format_exc()
                    }
            traceback.print_exc()
            continue

        result[series_no]['info'] = {
                'series_no': series_no,
                'description': rec_data['description'],
                'N_pixel_i': rec_data['N_pixel_i'],
                'N_pixel_j': rec_data['N_pixel_j'],
                'N_slice': len(rec_data['index_list']),
                'slice_thickness': rec_data['slice_thickness'],
                'spacing_i': rec_data['spacing_i'],
                'spacing_j': rec_data['spacing_j'],
                'spacing_slice': rec_data['spacing_slice'],
                'orient': rec_data['ImageOrientationPatient'],
                'orient_like': rec_data['series_like'],
                'patient_position': rec_data['patient_position'],
                'use_contrast': rec_data['use_contrast'],
                'chirality': rec_data['chirality'],
                }

        for inst in inst_list:
            inst_no = inst['instance_no']
            info = get_instance_info(inst['dcmObj'])
            result[series_no]['objs'][inst_no] = {
                    'fname': inst['fname'],
                    'dcmObj': inst['dcmObj'],
                    'image_arr': inst['image_arr'],
                    'info': {
                        'bits_origin': info['bits_origin'],
                        'image_position': info['image_position']
                        }
                    }
    return result


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
