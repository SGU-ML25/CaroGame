"""Các thao tác xử lý nhiều nút trong mỗi lượt mô phỏng PUCT."""

import cython
from libc.math cimport INFINITY, sqrt


cdef class PUCTNode:
    cdef public Py_ssize_t N, vl
    cdef public double W, prior
    cdef public object move
    cdef public list children

    def __cinit__(self, move, double prior):
        self.move = move
        self.prior = prior
        self.children = []

    @property
    def expanded(self):
        return bool(self.children)


@cython.cdivision(True)
cpdef PUCTNode select_child(PUCTNode node, double c_puct):
    cdef Py_ssize_t visits = node.N + node.vl
    cdef double sqrt_n = sqrt(visits if visits >= 1 else 1)
    cdef double q, score, best_score = -INFINITY
    cdef PUCTNode child, best = node.children[0]
    for child in node.children:
        visits = child.N + child.vl
        q = -(child.W + child.vl) / visits if visits else 0.0
        score = q + c_puct * child.prior * sqrt_n / (1 + visits)
        if score > best_score:
            best, best_score = child, score
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list make_children(const Py_ssize_t[::1] cells, const double[::1] priors):
    cdef Py_ssize_t i, n = cells.shape[0]
    if priors.shape[0] != n:
        raise ValueError("Mỗi ô hợp lệ cần đúng một prior.")
    return [PUCTNode(cells[i], priors[i]) for i in range(n)]
