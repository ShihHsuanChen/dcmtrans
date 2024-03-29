import numpy as np


def do_nothing(image: np.ndarray, *args, **kwargs):
    return image


def linear_trans(
        image: np.ndarray,
        intercept: float,
        slope: float,
        ) -> np.ndarray:
    return image * slope + intercept


def window_linear_trans(
        image: np.ndarray,
        window_center: float,
        window_width: float,
        depth: int = 256,
        ) -> np.ndarray:
    dtype = image.dtype
    image = image.astype(np.float64)
    a = image <= (window_center - 0.5 - (window_width - 1) / 2)
    b = image > window_center - 0.5 + (window_width - 1) / 2
    c = ((image - (window_center - 0.5)) / (window_width - 1) + 0.5) * (depth - 1)
    image = np.where(a, 0, np.where(b, depth-1, c))
    return image.clip(0, depth-1).astype(dtype)


def window_linear_exact_trans(
        image: np.ndarray,
        window_center: float,
        window_width: float,
        depth: int = 256,
        ) -> np.ndarray:
    dtype = image.dtype
    image = image.astype(np.float64)
    a = image <= (window_center - window_width / 2)
    b = image > (window_center + window_width / 2)
    c = (image - window_center) / window_width * (depth - 1)
    image = np.where(a, 0, np.where(b, depth-1, c))
    return image.clip(0, depth-1).astype(dtype)


def window_sigmoid_trans(image: np.ndarray, window_center: float, window_width: float, depth: int = 256) -> np.ndarray:
    return (depth - 1) / (1 + np.exp(-4*(image - window_center)/window_width))


def lut_trans(image, lut_descriptor, lut_data, scale_factor: float = 1) -> np.ndarray:
    lut_data = lut_data * scale_factor
    image = image - lut_descriptor[1]
    image[image <= 0] = 0
    image[image >= lut_descriptor[0]] = int(lut_descriptor[0] - 1)
    image = image.astype(int)
    return lut_data[image]
