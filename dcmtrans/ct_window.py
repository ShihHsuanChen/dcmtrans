from typing import Union
# preset window in Hounsfield Unit (HU)
PRESET_WINDOW = {
        'abdomen': {'wl': 60, 'ww': 400},
        'angio': {'wl': 300, 'ww': 600},
        'bone': {'wl': 300, 'ww': 1500},
        'brain': {'wl': 40, 'ww': 80},
        'mediastinum': {'wl': 40, 'ww': 400},
        'lung': {'wl': -750, 'ww': 1500}
        }
DEFULT = {'wl': 0, 'ww': 2000}


def get_window(body_part: str) -> [Union[float, int], Union[float, int]]:
    """

    :param body_part:
            body part or material name in low case.
            Valid values: abdomen, angio, bone, brain, mediastinum, lung
            otherwise, the default window level (0) window width (2000) will be given
    :return: window_level, window_width
        window_level: float
            window level (window center) in Hounsfield Unit (HU)
        window_width: float, int
            window width (full range, maximum - minimum) in Hounsfield Unit (HU)
    """
    w = PRESET_WINDOW.get(body_part)
    if w is None:
        w = DEFULT
    return w.get('wl'), w.get('ww')
