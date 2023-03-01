from pathlib import Path
from typing import TypeVar, Generic, Type, Tuple, Union


T = TypeVar('T')


def NTuple(T: Type, N: int):
    return Tuple[tuple([T]*N)]


PathLike = Union[str, Path]
