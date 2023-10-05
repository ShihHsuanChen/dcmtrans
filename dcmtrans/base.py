import abc
import warnings
from typing import Dict, Callable, Tuple, Optional

import pydicom


class BaseTransManager(abc.ABC):
    collection: Dict[str, Callable]
    retired: Dict[str, bool]
    experimental: Dict[str, bool]

    def __init__(self):
        self.collection = {}
        self.retired = {}
        self.experimental = {}

    def register(self,
            mode: str,
            retired: bool = False,
            skip_if_exists: bool = False,
            experimental: bool = False,
            ):
        def wrap_func(func: Callable):
            if mode in self.collection:
                if skip_if_exists:
                    warnings.warn(f'Mode "{mode}" was already registered. Skip')
                    return func
                warnings.warn(f'Mode "{mode}" was already registered. Overwrite')

            self.collection[mode] = func
            self.retired[mode] = retired
            self.experimental[mode] = experimental
            return func
        return wrap_func

    @abc.abstractmethod
    def get_mode(self, dcmObj: pydicom.FileDataset) -> Optional[str]:
        raise NotImplementedError()

    def get_func(self, dcmObj: pydicom.FileDataset) -> Tuple[str, Optional[Callable]]:
        mode = self.get_mode(dcmObj)
        func = self.collection.get(mode)
        if func is None:
            warnings.warn(f'Mode "{mode}" is not implemented.')
        if self.retired.get(mode):
            warnings.warn(f'Mode "{mode}" is retired.')
        if self.experimental.get(mode):
            warnings.warn(f'Implementation of "{mode}" mode is experimental, no test yet.')
        return mode, func

    @property
    def modes(self):
        return list(sorted(self.collection.keys()))
