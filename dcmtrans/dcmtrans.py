import numpy
import pydicom
import traceback
import PIL.Image

from typing import Union, Callable, Dict, List, Any

from .trans_method import do_nothing

from .voi_trans import voi_classifier
from .voi_trans import voi_linear_trans
from .voi_trans import voi_linear_exact_trans
from .voi_trans import voi_sigmoid_trans
from .voi_trans import voi_lut_trans

from .modality_trans import modality_classifier
from .modality_trans import modality_linear_trans
from .modality_trans import modality_lut_trans

from .pi_trans import pi_classifier
from .pi_trans import pi_monochrome1
from .pi_trans import pi_monochrome2

from .reconstruction import reconstruct_series
from .reconstruction import get_instance_info

from .formats import get_nbits_from_colormap


DEFULT_COLORMAP = 'L'
BITS_TRANS = get_nbits_from_colormap(DEFULT_COLORMAP)
DEPTH = int(2**BITS_TRANS)


def dcmtrans(
        dcmObj: pydicom.FileDataset,
        image_data: numpy.ndarray,
        depth: int=DEPTH,
        window: List[Union[str, Dict[str, Union[int, float]]]]=('default',)
        ) -> [List[Union[None, numpy.ndarray]], List[Union[None, Exception]], Dict[str, str]]:
    """

    :param dcmObj: object read from pydicom
    :param image_data: image array
    :param depth: output image bit-depth. example: 256 (2**8)
    :param window: given windows. [<window str, dict>]
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

    for w in window:
        try:
            m_image_data = numpy.copy(t_image_data)
            voi_mode, m_image_data = voi_trans(dcmObj, m_image_data, depth, unit=unit, window=w)
            pi_mode, m_image_data = pi_trans(dcmObj, m_image_data, depth)
            exc = None
        except Exception as e:
            m_image_data = None
            exc = e
        exception_list.append(exc)
        image_data_list.append(m_image_data)

    return image_data_list, exception_list, {'modality': mod_mode, 'modality_unit': unit, 'voi': voi_mode, 'pi': pi_mode}


def get_modality_trans_func(dcmObj: pydicom.FileDataset) -> [str, Union[Callable, None]]:
    mode = modality_classifier(dcmObj)

    if mode == 'LINEAR':
        return mode, modality_linear_trans
    elif mode == 'TABLE':
        return mode, modality_lut_trans
    else:
        return mode, None


def modality_trans(
        dcmObj: pydicom.FileDataset,
        image_data: numpy.ndarray,
        ) -> [str, numpy.ndarray, str]:
    mode, func = get_modality_trans_func(dcmObj)
    if func is not None:
        image_data, unit = func(dcmObj, image_data)
        return mode, image_data, unit
    else:
        return mode, do_nothing(image_data), None


def get_voi_trans_func(dcmObj: pydicom.FileDataset) -> [str, Union[Callable, None]]:
    mode = voi_classifier(dcmObj)

    if mode == 'LINEAR':
        return mode, voi_linear_trans
    elif mode == 'LINEAR_EXACT':
        return mode, voi_linear_exact_trans
    elif mode == 'SIGMOID':
        return mode, voi_sigmoid_trans
    elif mode == 'TABLE':
        return mode, voi_lut_trans
    else:
        return mode, None


def voi_trans(
        dcmObj: pydicom.FileDataset,
        image_data: numpy.ndarray,
        depth: int,
        unit: str=None,
        window='default'
        ) -> [str, numpy.ndarray]:
    mode, func = get_voi_trans_func(dcmObj)
    if func is not None:
        return mode, func(dcmObj, image_data, depth, window=window, unit=unit)
    else:
        return mode, do_nothing(image_data)


def get_pi_trans_func(dcmObj: pydicom.FileDataset) -> [str, Union[Callable, None]]:
    mode = pi_classifier(dcmObj)

    if mode == 'MONOCHROME1':
        return mode, pi_monochrome1
    elif mode == 'MONOCHROME2':
        return mode, pi_monochrome2
    else:
        return mode, None


def pi_trans(
        dcmObj: pydicom.FileDataset,
        image_data: numpy.ndarray,
        depth: int
        ) -> [str, numpy.ndarray]:
    mode, func = get_pi_trans_func(dcmObj)
    if func is not None:
        return mode, func(dcmObj, image_data, depth)
    else:
        return mode, do_nothing(image_data)


def reconstruct(tank: List[Dict[str, Any]]) -> Dict:
    """
    :param tank:
        [{
            series_no: <series_no>
            instance_no: <instance_no>
            dcmObj: <dicom object (FileDataset)>
            image_arr: <image array (np.ndarray) not important>
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
                        image_arr: <image array (np.ndarray) not important>
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


def image_convert(image_arr: numpy.ndarray, fmt='L') -> PIL.Image.Image:
    """

    :param image_arr: image data
    :param fmt: image format code
    :return: PIL.Image object
    """
    image = PIL.Image.fromarray(image_arr).convert(fmt)
    return image
