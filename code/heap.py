from typing import List

from util import swap


class Heap:
    def __init__(self, nodes: List[int] = None, d: int = 2) -> None:
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

        self.heap_size = len(self.nodes)
        self.d = d

    def __getitem__(self, idx: int) -> int:
        return self.nodes[idx]

    def __setitem__(self, idx: int, value: int) -> None:
        self.nodes[idx] = value


def nth_child(i: int, n: int, d: int = 2) -> int:
    """Big-Theta(1)"""
    return d*i + n + 1


def max_heapify(A: Heap, i: int = 0) -> None:
    """O(d*log_d(n))"""
    largest = i

    # Loop is Big-Theta(d)
    for j in range(A.d):  # executes d times
        child_idx = nth_child(i, j, A.d)   # constant time
        if child_idx < A.heap_size and A[child_idx] > A[largest]:
            largest = nth_child(i, j, A.d)

    if largest != i:
        swap(A, i, largest)  # constant time
        max_heapify(A, largest)


def extract_max(A: Heap) -> int:
    max_val = A[0]
    last = A[-1]
    A[0] = last
    A.heap_size -= 1

    max_heapify(A, 0)

    return max_val


def build_max_heap(A: Heap):
    """O(n*d*log_d(n))
    When n ~ d: O(n^2)
    """
    for i in range(A.heap_size // 2 - 1, -1, -1):
        max_heapify(A, i)


def parent(i: int, d: int = 2) -> int:
    return (i - 1) // d


def increase_key(A: Heap, x: int, k: float) -> None:
    if k < x.key:
        raise ValueError(f"New key {k} is smaller than current key {x.key}")

    x.key = k
    i = A.nodes.index(x)

    while i > 1 and A[parent(i, A.d)] < A[i]:
        A[i] = A[parent(i, A.d)]
        A[parent(i, A.d)] = x

        i = parent(i, A.d)


def is_max_heap(A: Heap):
    for i in range(1, A.heap_size):
        parent_idx = parent(i, A.d)

        if A[parent_idx] < A[i]:
            return False

    return True


def heap_sort(A: List[int], d: int = 2, copy: bool = False) -> List[int]:
    """O(n*d*log_d(n))"""
    if copy:
        A = A.copy()

    sort_heap = Heap(A, d)
    build_max_heap(sort_heap)  # O(n*d*log_d(n))

    for i in range(sort_heap.heap_size - 1, -1, -1):
        swap(sort_heap, 0, i)
        sort_heap.heap_size -= 1
        max_heapify(sort_heap, 0) # O(d*log_d(n))

    return sort_heap.nodes
