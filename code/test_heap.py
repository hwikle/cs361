from random import randint

import heap
from util import rand_list, is_sorted


def test_is_max_heap():
    node_list = [5, 4, 3]
    A = heap.Heap(node_list)
    assert heap.is_max_heap(A)


def test_is_max_heap_negative_case():
    node_list = [3, 4, 5]
    A = heap.Heap(node_list)
    assert not heap.is_max_heap(A)


def test_parents_are_correct():
    parents = [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    correct = True

    for i, p in enumerate(parents):
        if heap.parent(i, 3) != p:
            correct = False
            break

    assert correct


def test_nth_child():
    children = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    d = 3

    correct = True

    for i in range(4):
        for j in range(d):
            if heap.nth_child(i, j, d) != children[i][j]:
                correct = False
                break

    assert correct


def test_max_heapify():
    node_list = [3, 4, 5]
    A = heap.Heap(node_list)

    heap.max_heapify(A)
    assert heap.is_max_heap(A)


def test_build_max_heap():
    node_list = rand_list(100)
    d = randint(2, 10)
    A = heap.Heap(node_list, d)
    heap.build_max_heap(A)

    assert heap.is_max_heap(A)


def test_extract_max_preserves_heap_property():
    node_list = rand_list(100)
    A = heap.Heap(node_list)
    heap.build_max_heap(A)

    heap.extract_max(A)

    assert heap.is_max_heap(A)


def test_heap_sort_correctness():
    r_list = rand_list(100)

    correct = True

    for i in range(10):
        d = randint(2, 10)

        if not is_sorted(heap.heap_sort(r_list, d)):
            correct = False
            break

    assert correct
