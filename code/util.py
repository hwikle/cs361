from random import randint
from typing import Protocol, TypeVar, List

T = TypeVar('T')


class GetableSetable(Protocol[T]):
    def __getitem__(self, item: int) -> T:
        ...

    def __setitem__(self, key: int, value: T) -> None:
        ...


def swap(a: GetableSetable, idx1: int, idx2: int) -> None:
    tmp = a[idx1]
    a[idx1] = a[idx2]
    a[idx2] = tmp


def rand_list(n: int, max_int: int = 100) -> List[int]:
    rlist = []

    for i in range(n):
        rlist.append(randint(0, max_int))

    return rlist


def is_sorted(l: List[int]) -> bool:
    return l == sorted(l)
