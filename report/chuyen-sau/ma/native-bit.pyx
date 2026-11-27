cdef inline bint has_five_c(u128 x) noexcept nogil:
    cdef int i, d
    cdef u128 t
    for i in range(4):
        d = D[i]
        t = x & (x >> d)
        t &= t >> (2 * d)
        t &= x >> (4 * d)
        if t & START[i]:
            return True
    return False


cdef inline u128 dilate(u128 x) noexcept nogil:
    cdef u128 v = x | (x << N) | (x >> N)
    cdef u128 h = v | ((v << 1) & NOT_COL0) | ((v >> 1) & NOT_COL10)
    return h & FULL


cdef inline u128 candidates_c(u128 me, u128 opp, int radius) noexcept nogil:
    cdef u128 occ = me | opp
    cdef u128 area
    cdef int r
    if occ == 0:
        return (<u128>1) << CENTER
    area = occ
    for r in range(radius):
        area = dilate(area)
    return area & ~occ & FULL


cdef inline void window_counts(u128 me, u128 opp, int i, u128* c) noexcept nogil:
    """c[0..3] = c0, c1, c2, free (giống window_counts của bản Python)."""
    cdef int d = D[i]
    cdef int k
    cdef u128 free = START[i]
    cdef u128 c0 = 0, c1 = 0, c2 = 0, p, carry, carry2
    for k in range(WIN):
        free &= ~(opp >> (k * d))
    for k in range(WIN):
        p = (me >> (k * d)) & free
        carry = c0 & p
        c0 ^= p
        carry2 = c1 & carry
        c1 ^= carry
        c2 |= carry2
    c[0] = c0
    c[1] = c1
    c[2] = c2
    c[3] = free


cdef inline u128 exactly(int k, u128* c) noexcept nogil:
    cdef u128 m = c[3]
    m &= c[0] if (k & 1) else ~c[0]
    m &= c[1] if (k & 2) else ~c[1]
    m &= c[2] if (k & 4) else ~c[2]
    return m


cdef inline u128 winning_squares_c(u128 me, u128 opp) noexcept nogil:
    cdef u128 empty = ~(me | opp) & FULL
    cdef u128 res = 0, w4
    cdef u128 c[4]
    cdef int i, j
    for i in range(4):
        window_counts(me, opp, i, c)
        w4 = exactly(4, c)
        for j in range(WIN):
            res |= (w4 << (j * D[i])) & empty
    return res
