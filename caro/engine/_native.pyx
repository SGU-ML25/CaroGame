# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Lõi native (Cython) của cấp 3: bitboard 128 bit + negamax Alpha-Beta.

Bản dịch từng dòng của caro/engine/bitboard.py và AlphaBetaAgent._negamax / _order_moves / _forced
(caro/agents/alphabeta.py): cùng thứ tự duyệt nước, cùng cách xếp nước, cùng cách thay thế trong bảng chuyển vị,
nên với cùng tham số (không tính giới hạn thời gian) cho cùng điểm, cùng PV và cùng số nút như bản Python.
tests/test_native.py kiểm tra điều đó.

Bàn cờ 121 bit nằm trong `unsigned __int128` (gcc / clang). Build: `python setup_native.py build_ext --inplace`.
Nếu không build được thì AlphaBetaAgent vẫn chạy bằng bản Python (backend="python" hoặc "auto").
"""

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint64_t
from libcpp.deque cimport deque
from libcpp.unordered_map cimport unordered_map
from posix.time cimport CLOCK_MONOTONIC, clock_gettime, timespec

cdef extern from *:
    """
    typedef unsigned __int128 u128_t;
    """
    ctypedef unsigned long long u128 "u128_t"

cdef extern from *:
    int popcnt64 "__builtin_popcountll"(uint64_t) nogil
    int ctz64 "__builtin_ctzll"(uint64_t) nogil

# ---------------------------------------------------------------------------
# Hằng số (lấy từ bản Python để chắc chắn giống hệt)
# ---------------------------------------------------------------------------

DEF N = 11
DEF CELLS = 121
DEF WIN = 5
DEF MAXPLY = 100            # độ sâu tối đa (ply) mà bộ đệm PV chứa được
DEF KILLERN = 64            # giống len(self._killer) của bản Python
DEF PVN = 102               # MAXPLY + 2: kích thước bộ đệm PV (viết thành hằng để trình kiểm tra cú pháp không báo lỗi)

cdef int64_t INF = 10_000_000
cdef int64_t WIN_SCORE = 1_000_000

cdef int EXACT = 0
cdef int LOWER = 1
cdef int UPPER = 2

cdef int D[4]
cdef u128 START[4]
cdef u128 FULL
cdef u128 NOT_COL0
cdef u128 NOT_COL10
cdef int CENTER
cdef u128 WINAT[CELLS][4]     # ô -> với mỗi hướng, các ô đầu của những cửa sổ chứa ô đó
cdef uint64_t ZOB[2][CELLS]
cdef uint64_t SIDEKEY


cdef inline u128 to128(object x):
    return ((<u128><uint64_t>(x >> 64)) << 64) | (<u128><uint64_t>(x & 0xFFFFFFFFFFFFFFFF))


cdef inline object from128(u128 v):
    return ((<object><uint64_t>(v >> 64)) << 64) | (<object><uint64_t>v)


cdef void _init_tables():
    from caro.engine import bitboard as bb
    from caro.engine.zobrist import SIDE_KEY, ZOBRIST

    global FULL, NOT_COL0, NOT_COL10, CENTER, SIDEKEY
    FULL = to128(bb.FULL)
    NOT_COL0 = to128(bb.NOT_COL0)
    NOT_COL10 = to128(bb.NOT_COL10)
    CENTER = bb.CENTER
    SIDEKEY = SIDE_KEY
    for i, (d, start) in enumerate(bb.LINES):
        D[i] = d
        START[i] = to128(start)
    for m in range(CELLS):
        for i, (d, start) in enumerate(bb.LINES):
            w = 0
            for j in range(WIN):
                if m - j * d >= 0 and (start >> (m - j * d)) & 1:
                    w |= 1 << (m - j * d)
            WINAT[m][i] = to128(w)
    for s in range(2):
        for m in range(CELLS):
            ZOB[s][m] = ZOBRIST[s][m]


_init_tables()

# ---------------------------------------------------------------------------
# Bitboard
# ---------------------------------------------------------------------------

cdef inline int pop(u128 x) noexcept nogil:
    return popcnt64(<uint64_t>x) + popcnt64(<uint64_t>(x >> 64))


cdef inline int lowest(u128 x) noexcept nogil:
    cdef uint64_t lo = <uint64_t>x
    if lo:
        return ctz64(lo)
    return 64 + ctz64(<uint64_t>(x >> 64))


cdef inline int bits_of(u128 x, int* out) noexcept nogil:
    """Chỉ số các bit 1 theo thứ tự tăng dần (giống iter_bits)."""
    cdef int n = 0
    cdef int i
    while x:
        i = lowest(x)
        out[n] = i
        n += 1
        x &= x - 1
    return n


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


cdef inline int64_t side_score_c(u128 me, u128 opp, int64_t* w) noexcept nogil:
    cdef int64_t s = 0
    cdef u128 c[4]
    cdef int i, k
    for i in range(4):
        window_counts(me, opp, i, c)
        for k in range(1, WIN):
            s += w[k] * pop(exactly(k, c))
    return s


# --- API Python để kiểm thử đối chiếu với caro/engine/bitboard.py ---

def has_five(x):
    return has_five_c(to128(x))


def candidates(me, opp, radius=2):
    return from128(candidates_c(to128(me), to128(opp), radius))


def winning_squares(me, opp):
    return from128(winning_squares_c(to128(me), to128(opp)))


def side_score(me, opp, weight):
    cdef int64_t w[5]
    for i in range(5):
        w[i] = weight[i]
    return side_score_c(to128(me), to128(opp), w)


def evaluate(me, opp, weight):
    cdef int64_t w[5]
    for i in range(5):
        w[i] = weight[i]
    return side_score_c(to128(me), to128(opp), w) - side_score_c(to128(opp), to128(me), w)


# ---------------------------------------------------------------------------
# Thông tin đe doạ (giống _threat_info)
# ---------------------------------------------------------------------------

cdef struct Threat:
    u128 wins
    bint has_four
    u128 gain[CELLS]        # 0 = ô không có trong dict `gain` của bản Python
    u128 c[4][4]


cdef void threat_info(u128 me, u128 opp, Threat* t) noexcept nogil:
    cdef u128 empty = ~(me | opp) & FULL
    cdef u128 w4, w3
    cdef int i, j, s, d, idx, a, b, found, n3
    cdef int s3[CELLS]
    for i in range(CELLS):
        t.gain[i] = 0
    t.wins = 0
    t.has_four = False
    for i in range(4):
        d = D[i]
        window_counts(me, opp, i, t.c[i])
        w4 = exactly(4, t.c[i])
        if w4:
            t.has_four = True
            for j in range(WIN):
                t.wins |= (w4 << (j * d)) & empty
        w3 = exactly(3, t.c[i])
        n3 = bits_of(w3, s3)
        for j in range(n3):
            s = s3[j]
            a = -1
            b = -1
            found = 0
            for idx in range(WIN):
                if not ((me >> (s + idx * d)) & 1):
                    if found == 0:
                        a = s + idx * d
                    else:
                        b = s + idx * d
                    found += 1
            t.gain[a] |= (<u128>1) << b
            t.gain[b] |= (<u128>1) << a


# ---------------------------------------------------------------------------
# Bảng chuyển vị (giống TranspositionTable: replace-always, đẩy khoá cũ nhất ra khi đầy)
# ---------------------------------------------------------------------------

cdef struct TTE:
    int64_t value
    int depth
    int flag
    int move


cdef inline double now_s() noexcept nogil:
    cdef timespec ts
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return <double>ts.tv_sec + <double>ts.tv_nsec * 1e-9


cdef class Native:
    cdef bint use_tt, pruning, move_ordering, forced_moves, no_defense, has_deadline
    cdef int radius
    cdef int64_t w[5]
    cdef double deadline_end
    cdef int64_t nodes
    cdef size_t tt_size
    cdef unordered_map[uint64_t, TTE] tt
    cdef deque[uint64_t] tt_fifo
    cdef int killer[KILLERN][2]
    cdef int64_t hist[CELLS]
    cdef bint hist_has[CELLS]
    cdef int pv[PVN][PVN]
    cdef int pvlen[PVN]

    def __init__(self, bint use_tt, bint pruning, bint move_ordering, bint forced_moves, int radius,
                 bint no_defense, weight, int tt_size_mb=64):
        self.use_tt = use_tt
        self.pruning = pruning
        self.move_ordering = move_ordering
        self.forced_moves = forced_moves
        self.radius = radius
        self.no_defense = no_defense
        for i in range(5):
            self.w[i] = weight[i]
        self.tt_size = max(1, (tt_size_mb * 1024 * 1024) // 200)
        self.has_deadline = False
        self.deadline_end = 0.0
        self.nodes = 0
        self.reset_search()

    # --- trạng thái ---------------------------------------------------

    def reset_search(self):
        """Xoá killer và history (tương đương self._killer = ...; self._history.clear())."""
        cdef int i
        for i in range(KILLERN):
            self.killer[i][0] = -1
            self.killer[i][1] = -1
        for i in range(CELLS):
            self.hist[i] = 0
            self.hist_has[i] = False

    def clear_tt(self):
        self.tt.clear()
        self.tt_fifo.clear()

    def set_deadline(self, end):
        """`end` là mốc time.perf_counter() (giây) hoặc None = không giới hạn."""
        if end is None:
            self.has_deadline = False
        else:
            self.has_deadline = True
            self.deadline_end = end

    def take_nodes(self):
        cdef int64_t n = self.nodes
        self.nodes = 0
        return n

    @property
    def tt_len(self):
        return self.tt.size()

    cdef inline bint expired(self) noexcept nogil:
        return self.has_deadline and now_s() >= self.deadline_end

    cdef inline void tt_put(self, uint64_t key, int64_t value, int depth, int flag, int move) noexcept nogil:
        cdef TTE e
        cdef uint64_t oldest
        cdef unordered_map[uint64_t, TTE].iterator it
        if self.tt.size() >= self.tt_size:
            oldest = self.tt_fifo.front()
            self.tt_fifo.pop_front()
            self.tt.erase(oldest)
        e.value = value
        e.depth = depth
        e.flag = flag
        e.move = move
        it = self.tt.find(key)
        if it == self.tt.end():
            self.tt_fifo.push_back(key)
        self.tt[key] = e

    cdef inline int64_t eval_leaf(self, u128 me, u128 opp) noexcept nogil:
        if self.no_defense:
            return side_score_c(me, opp, self.w)
        return side_score_c(me, opp, self.w) - side_score_c(opp, me, self.w)

    # --- sắp xếp nước (giống _order_moves) -----------------------------

    cdef void order_moves(self, int* mv, int n, u128 me, u128 opp, uint64_t key, int ply) noexcept nogil:
        if (not self.move_ordering) or n == 1:
            return
        cdef int tt_move = -1
        cdef unordered_map[uint64_t, TTE].iterator it
        cdef Threat mine, theirs
        cdef int k0 = -1, k1 = -1
        cdef int i, j, m, a, kk
        cdef int64_t s, base = 0
        cdef u128 bit_m, windows
        cdef bint position_ready = False
        cdef int nterms = 0
        cdef int tdir[20]
        cdef u128 twin[20]
        cdef int64_t tdelta[20]
        cdef int64_t before, after
        cdef int64_t sc[CELLS]
        cdef int tmp_m
        cdef int64_t tmp_s

        if self.use_tt:
            it = self.tt.find(key)
            if it != self.tt.end():
                tt_move = deref(it).second.move
        threat_info(me, opp, &mine)
        threat_info(opp, me, &theirs)
        if ply < KILLERN:
            k0 = self.killer[ply][0]
            k1 = self.killer[ply][1]

        for i in range(n):
            m = mv[i]
            bit_m = (<u128>1) << m
            if m == tt_move:
                s = 10_000_000
            elif bit_m & mine.wins:
                s = 9_500_000
            elif pop(mine.wins | mine.gain[m]) >= 2:
                s = 9_000_000
            elif mine.has_four or mine.gain[m] != 0:
                s = 8_000_000
            elif pop((theirs.wins & ~bit_m) | theirs.gain[m]) >= 2:
                s = 7_000_000
            elif m == k0:
                s = 6_000_000
            elif m == k1:
                s = 5_000_000
            elif self.hist_has[m]:
                s = 1_000_000 + self.hist[m]
            else:
                if not position_ready:
                    position_ready = True
                    for a in range(4):
                        for kk in range(WIN):
                            windows = exactly(kk, mine.c[a])
                            if not windows:
                                continue
                            before = self.w[kk] if kk else 0
                            after = self.w[kk + 1] if kk + 1 < WIN else 0
                            base += before * pop(windows)
                            if after != before:
                                tdir[nterms] = a
                                twin[nterms] = windows
                                tdelta[nterms] = after - before
                                nterms += 1
                s = base
                for j in range(nterms):
                    s += tdelta[j] * pop(twin[j] & WINAT[m][tdir[j]])
            sc[i] = s

        # sort giảm dần theo (điểm, ô) như scores.sort(reverse=True) trên tuple (s, m)
        for i in range(1, n):
            tmp_m = mv[i]
            tmp_s = sc[i]
            j = i - 1
            while j >= 0 and (sc[j] < tmp_s or (sc[j] == tmp_s and mv[j] < tmp_m)):
                mv[j + 1] = mv[j]
                sc[j + 1] = sc[j]
                j -= 1
            mv[j + 1] = tmp_m
            sc[j + 1] = tmp_s

    # --- negamax (giống _negamax) --------------------------------------

    cdef int64_t nm(self, u128 me, u128 opp, uint64_t key, int side, int depth,
                    int64_t alpha, int64_t beta, int ply) noexcept nogil:
        cdef unordered_map[uint64_t, TTE].iterator it
        cdef TTE e
        cdef int mv[CELLS]
        cdef int n, i, m, best_m, k, flag
        cdef u128 cand, win_sqs, opp_win, forced
        cdef int64_t orig_alpha, best, score
        cdef u128 new_opp
        cdef uint64_t new_key

        self.nodes += 1
        self.pvlen[ply] = 0

        if has_five_c(opp):
            return -(WIN_SCORE - ply)

        if self.use_tt:
            it = self.tt.find(key)
            if it != self.tt.end():
                e = deref(it).second
                if e.depth >= depth:
                    if e.flag == EXACT:
                        if e.move >= 0:
                            self.pv[ply][0] = e.move
                            self.pvlen[ply] = 1
                        return e.value
                    elif e.flag == LOWER:
                        alpha = max(alpha, e.value)
                    elif e.flag == UPPER:
                        beta = min(beta, e.value)
                    if self.pruning and alpha >= beta:
                        if e.move >= 0:
                            self.pv[ply][0] = e.move
                            self.pvlen[ply] = 1
                        return e.value

        if depth == 0 or self.expired():
            return self.eval_leaf(me, opp)

        cand = candidates_c(me, opp, self.radius)
        n = bits_of(cand, mv)
        if n == 0:
            return self.eval_leaf(me, opp)

        if self.forced_moves:
            win_sqs = winning_squares_c(me, opp)
            forced = cand & win_sqs
            if not forced:
                opp_win = winning_squares_c(opp, me)
                forced = cand & opp_win
            if forced:
                n = bits_of(forced, mv)

        self.order_moves(mv, n, me, opp, key, ply)

        orig_alpha = alpha
        best = -INF
        best_m = mv[0]

        for i in range(n):
            m = mv[i]
            new_opp = me | ((<u128>1) << m)
            new_key = key ^ ZOB[side][m] ^ SIDEKEY
            score = -self.nm(opp, new_opp, new_key, side ^ 1, depth - 1, -beta, -alpha, ply + 1)

            if score > best:
                best = score
                best_m = m
                self.pv[ply][0] = m
                for k in range(self.pvlen[ply + 1]):
                    self.pv[ply][k + 1] = self.pv[ply + 1][k]
                self.pvlen[ply] = 1 + self.pvlen[ply + 1]
            if score > alpha:
                alpha = score
            if self.pruning and alpha >= beta:
                if ply < KILLERN and m != self.killer[ply][0]:
                    self.killer[ply][1] = self.killer[ply][0]
                    self.killer[ply][0] = m
                self.hist[m] += <int64_t>depth * depth
                self.hist_has[m] = True
                break
            if self.expired():
                break

        if self.use_tt and not self.expired():
            if orig_alpha < best and best < beta:
                flag = EXACT
            elif best >= beta:
                flag = LOWER
            else:
                flag = UPPER
            self.tt_put(key, best, depth, flag, best_m)

        return best

    # --- API cho AlphaBetaAgent ----------------------------------------

    def negamax(self, me, opp, key, int side, int depth, int64_t alpha, int64_t beta, int ply):
        """Tương đương AlphaBetaAgent._negamax (không kèm deadline: gọi set_deadline trước). Trả (điểm, pv)."""
        if depth + ply >= MAXPLY:
            raise ValueError("độ sâu vượt giới hạn của bản native")
        cdef int64_t score = self.nm(to128(me), to128(opp), <uint64_t>key, side, depth, alpha, beta, ply)
        return score, [self.pv[ply][i] for i in range(self.pvlen[ply])]

    def root(self, me, opp, key, int side, list moves, int depth):
        """Tương đương AlphaBetaAgent._negamax_root."""
        if depth >= MAXPLY:
            raise ValueError("độ sâu vượt giới hạn của bản native")
        cdef int mv[CELLS]
        cdef int n = len(moves)
        cdef int i, m, k
        cdef u128 m128 = to128(me)
        cdef u128 o128 = to128(opp)
        cdef uint64_t k64 = <uint64_t>key
        cdef int64_t alpha = -INF, beta = INF, best = -INF, score
        cdef list best_pv = []
        for i in range(n):
            mv[i] = moves[i]
        self.order_moves(mv, n, m128, o128, k64, 0)
        for i in range(n):
            m = mv[i]
            score = -self.nm(o128, m128 | ((<u128>1) << m), k64 ^ ZOB[side][m] ^ SIDEKEY, side ^ 1,
                             depth - 1, -beta, -alpha, 1)
            if score > best:
                best = score
                best_pv = [m] + [self.pv[1][k] for k in range(self.pvlen[1])]
            if score > alpha:
                alpha = score
            if self.pruning and alpha >= beta:
                break
            if self.expired():
                break
        return best, best_pv
