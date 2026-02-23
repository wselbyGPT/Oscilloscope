#!/usr/bin/env python3
import curses
import locale
import math
import time
import json
import copy
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import numpy as np


# =========================
# Helpers
# =========================
def safe_addstr(stdscr, y, x, s, attr=0):
    try:
        stdscr.addstr(y, x, s, attr)
    except curses.error:
        pass

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def wrap_pi(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

def deg(a_rad):
    return math.degrees(a_rad)

def fmt_deg(a_rad):
    d = ((deg(a_rad) + 180.0) % 360.0) - 180.0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:6.1f}°"

def cplx_phase(z: complex) -> float:
    return math.atan2(z.imag, z.real)

def basis_label(i: int, n: int) -> str:
    return "|" + format(i, f"0{n}b") + "⟩"


# =========================
# Color palettes
# =========================
def init_palettes(stdscr):
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    colors = getattr(curses, "COLORS", 0)
    max_pairs = getattr(curses, "COLOR_PAIRS", 0)
    pair = 1

    # Phase palette (12 hues)
    xterm12 = [196, 208, 226, 118, 46, 48, 51, 39, 21, 93, 201, 198]
    basic12 = [
        curses.COLOR_RED, curses.COLOR_YELLOW, curses.COLOR_YELLOW,
        curses.COLOR_GREEN, curses.COLOR_GREEN, curses.COLOR_CYAN,
        curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_BLUE,
        curses.COLOR_MAGENTA, curses.COLOR_MAGENTA, curses.COLOR_RED
    ]
    phase_pairs = []
    for k in range(12):
        if pair >= max_pairs:
            break
        fg = xterm12[k] if colors >= 256 else basic12[k]
        try:
            curses.init_pair(pair, fg, -1)
            phase_pairs.append(curses.color_pair(pair))
        except curses.error:
            phase_pairs.append(0)
        pair += 1
    if not phase_pairs:
        phase_pairs = [0]

    def phase_attr(angle_rad: float) -> int:
        a = wrap_pi(angle_rad)
        t = (a + math.pi) / (2 * math.pi)
        idx = int(t * len(phase_pairs)) % len(phase_pairs)
        return phase_pairs[idx]

    # Heat palette (8 levels)
    heat_pairs = []
    heat_xterm = [17, 19, 21, 33, 39, 45, 220, 196]  # blue→cyan→yellow→red
    heat_basic = [curses.COLOR_BLUE, curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW,
                  curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_RED]
    for k in range(8):
        if pair >= max_pairs:
            break
        fg = heat_xterm[k] if colors >= 256 else heat_basic[k]
        try:
            curses.init_pair(pair, fg, -1)
            heat_pairs.append(curses.color_pair(pair))
        except curses.error:
            heat_pairs.append(0)
        pair += 1
    if not heat_pairs:
        heat_pairs = [0]

    def heat_attr(intensity_0_1: float) -> int:
        t = clamp(intensity_0_1, 0.0, 1.0)
        idx = int(t * (len(heat_pairs) - 1) + 1e-9)
        return heat_pairs[idx]

    # Basic pairs
    def mk_pair(fg):
        nonlocal pair
        if pair >= max_pairs:
            pair += 1
            return 0
        try:
            curses.init_pair(pair, fg, -1)
            a = curses.color_pair(pair)
            pair += 1
            return a
        except curses.error:
            pair += 1
            return 0

    c_hdr = mk_pair(curses.COLOR_GREEN)
    c_met = mk_pair(curses.COLOR_MAGENTA)
    c_prob = mk_pair(curses.COLOR_CYAN)
    c_shot = mk_pair(curses.COLOR_YELLOW)
    c_err = mk_pair(curses.COLOR_RED)

    return phase_attr, heat_attr, c_hdr, c_met, c_prob, c_shot, c_err


# =========================
# Quantum gates + state ops
# =========================
def gate_I():
    return np.array([[1, 0], [0, 1]], dtype=np.complex128)

def gate_H():
    return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

def gate_X():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

def gate_Y():
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

def gate_Z():
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)

def gate_S():
    return np.array([[1, 0], [0, 1j]], dtype=np.complex128)

def gate_T():
    return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)

def gate_RX(theta: float):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

def gate_RY(theta: float):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def gate_RZ(theta: float):
    return np.array([
        [np.exp(-1j * theta / 2), 0j],
        [0j, np.exp(+1j * theta / 2)]
    ], dtype=np.complex128)

def apply_1q(state: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    out = state.copy()
    step = 1 << q
    block = step << 1
    dim = 1 << n
    for base in range(0, dim, block):
        for i in range(step):
            i0 = base + i
            i1 = i0 + step
            a0 = out[i0]
            a1 = out[i1]
            out[i0] = U[0, 0] * a0 + U[0, 1] * a1
            out[i1] = U[1, 0] * a0 + U[1, 1] * a1
    return out

def apply_cnot(state: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    if control == target:
        return state
    out = state.copy()
    dim = 1 << n
    c_mask = 1 << control
    t_mask = 1 << target
    for i in range(dim):
        if i & c_mask:
            j = i ^ t_mask
            if j > i:
                out[i], out[j] = out[j], out[i]
    return out

def apply_cz(state: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    if control == target:
        return state
    out = state.copy()
    dim = 1 << n
    cm = 1 << control
    tm = 1 << target
    for i in range(dim):
        if (i & cm) and (i & tm):
            out[i] *= -1.0
    return out

def apply_swap(state: np.ndarray, n: int, q1: int, q2: int) -> np.ndarray:
    if q1 == q2:
        return state
    out = state.copy()
    dim = 1 << n
    m1 = 1 << q1
    m2 = 1 << q2
    for i in range(dim):
        b1 = 1 if (i & m1) else 0
        b2 = 1 if (i & m2) else 0
        if b1 != b2:
            j = i ^ (m1 | m2)
            if j > i:
                out[i], out[j] = out[j], out[i]
    return out

def probs_from_state(psi: np.ndarray) -> np.ndarray:
    p = np.abs(psi) ** 2
    s = float(np.sum(p))
    if s > 0:
        p = p / s
    return p


# =========================
# Reduced rho + Bloch
# =========================
def reduced_rho_1q(psi: np.ndarray, n: int, k: int) -> np.ndarray:
    dim = 1 << n
    mask = 1 << k
    rho00 = 0j
    rho11 = 0j
    rho01 = 0j
    for i0 in range(dim):
        if i0 & mask:
            continue
        i1 = i0 | mask
        a0 = psi[i0]
        a1 = psi[i1]
        rho00 += a0 * np.conjugate(a0)
        rho11 += a1 * np.conjugate(a1)
        rho01 += a0 * np.conjugate(a1)
    rho = np.array([[rho00, rho01],
                    [np.conjugate(rho01), rho11]], dtype=np.complex128)
    tr = float(np.real(rho00 + rho11))
    if tr > 0:
        rho /= tr
    return rho

def bloch_from_rho(rho: np.ndarray):
    rho01 = rho[0, 1]
    x = 2.0 * float(np.real(rho01))
    y = 2.0 * float(np.imag(rho01))
    z = float(np.real(rho[0, 0] - rho[1, 1]))
    return clamp(x, -1.0, 1.0), clamp(y, -1.0, 1.0), clamp(z, -1.0, 1.0)

def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


# =========================
# Entanglement-ish metrics
# =========================
def exp_Zi(p: np.ndarray, n: int, i: int) -> float:
    dim = p.shape[0]
    mask = 1 << i
    s = 0.0
    for idx in range(dim):
        bit = 1 if (idx & mask) else 0
        z = 1.0 if bit == 0 else -1.0
        s += p[idx] * z
    return float(s)

def exp_ZiZj(p: np.ndarray, n: int, i: int, j: int) -> float:
    dim = p.shape[0]
    mi = 1 << i
    mj = 1 << j
    s = 0.0
    for idx in range(dim):
        bi = 1 if (idx & mi) else 0
        bj = 1 if (idx & mj) else 0
        zi = 1.0 if bi == 0 else -1.0
        zj = 1.0 if bj == 0 else -1.0
        s += p[idx] * (zi * zj)
    return float(s)

def corr_matrix_Z(p: np.ndarray, n: int):
    Ez = np.array([exp_Zi(p, n, i) for i in range(n)], dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0
                continue
            Eij = exp_ZiZj(p, n, i, j)
            cov = Eij - Ez[i] * Ez[j]
            var_i = max(1e-12, 1.0 - Ez[i] * Ez[i])
            var_j = max(1e-12, 1.0 - Ez[j] * Ez[j])
            r = cov / math.sqrt(var_i * var_j)
            M[i, j] = clamp(abs(r), 0.0, 1.0)
    return M

def mutual_information_Z(p: np.ndarray, n: int, i: int, j: int) -> float:
    dim = p.shape[0]
    mi = 1 << i
    mj = 1 << j
    P = np.zeros((2, 2), dtype=np.float64)
    for idx in range(dim):
        bi = 1 if (idx & mi) else 0
        bj = 1 if (idx & mj) else 0
        P[bi, bj] += p[idx]
    Pi = P.sum(axis=1)
    Pj = P.sum(axis=0)

    mi_bits = 0.0
    for bi in (0, 1):
        for bj in (0, 1):
            pij = P[bi, bj]
            if pij <= 0:
                continue
            denom = Pi[bi] * Pj[bj]
            if denom <= 0:
                continue
            mi_bits += pij * math.log(pij / denom, 2)
    return float(max(0.0, mi_bits))

def bell_fidelities(psi: np.ndarray):
    # n=2 only, ordering |00>,|01>,|10>,|11> with q0 LSB
    s2 = 1 / math.sqrt(2)
    bells = {
        "Phi+": np.array([s2, 0, 0, s2], dtype=np.complex128),
        "Phi-": np.array([s2, 0, 0, -s2], dtype=np.complex128),
        "Psi+": np.array([0, s2, s2, 0], dtype=np.complex128),
        "Psi-": np.array([0, s2, -s2, 0], dtype=np.complex128),
    }
    fids = {}
    for name, b in bells.items():
        ov = np.vdot(b, psi)
        fids[name] = float(abs(ov) ** 2)
    best = max(fids.items(), key=lambda kv: kv[1])
    return fids, best

def concurrence_pure_2q(psi: np.ndarray) -> float:
    a00, a01, a10, a11 = psi[0], psi[1], psi[2], psi[3]
    return float(clamp(2.0 * abs(a00 * a11 - a01 * a10), 0.0, 1.0))


# =========================
# Circuit model
# =========================
ONEQ_KINDS = {"H","X","Y","Z","S","T","RX","RY","RZ","MEAS"}
TWOQ_KINDS = {"CNOT","CZ","SWAP"}
SPAN_KINDS = {"ORACLE","DIFF"}  # optional (Grover templates)

@dataclass
class Op:
    kind: str
    q: Optional[int] = None
    control: Optional[int] = None
    target: Optional[int] = None
    q2: Optional[int] = None        # for SWAP (q and q2)
    theta: Optional[float] = None   # for rotations
    label: Optional[str] = None     # display override

    def touches(self) -> List[int]:
        qs = []
        if self.kind in ONEQ_KINDS and self.q is not None:
            qs.append(self.q)
        if self.kind == "CNOT" or self.kind == "CZ":
            if self.control is not None: qs.append(self.control)
            if self.target is not None: qs.append(self.target)
        if self.kind == "SWAP":
            if self.q is not None: qs.append(self.q)
            if self.q2 is not None: qs.append(self.q2)
        return qs

@dataclass
class Circuit:
    n: int
    columns: List[List[Op]]
    oracle_mark: int = 0  # used if ORACLE in circuit

    def to_dict(self):
        return {
            "n": self.n,
            "oracle_mark": int(self.oracle_mark),
            "columns": [[asdict(op) for op in col] for col in self.columns],
        }

    @staticmethod
    def from_dict(d):
        n = int(d.get("n", 1))
        oracle_mark = int(d.get("oracle_mark", 0))
        cols = []
        for col in d.get("columns", []):
            ops = []
            for op_d in col:
                ops.append(Op(**op_d))
            cols.append(ops)
        if not cols:
            cols = [[]]
        return Circuit(n=n, columns=cols, oracle_mark=oracle_mark)

    def ensure_col(self, c: int):
        while c >= len(self.columns):
            self.columns.append([])

    def insert_col(self, c: int):
        c = int(clamp(c, 0, len(self.columns)))
        self.columns.insert(c, [])

    def delete_col(self, c: int):
        if 0 <= c < len(self.columns):
            self.columns.pop(c)
        if not self.columns:
            self.columns = [[]]

    def remove_ops_touching(self, c: int, qubits: List[int]):
        if not (0 <= c < len(self.columns)):
            return
        qs = set(qubits)
        new_ops = []
        for op in self.columns[c]:
            if op.kind in SPAN_KINDS:
                # if placing something else, wipe span ops
                continue
            if any(q in qs for q in op.touches()):
                continue
            new_ops.append(op)
        self.columns[c] = new_ops

    def wipe_col(self, c: int):
        if 0 <= c < len(self.columns):
            self.columns[c] = []

    def place_oneq(self, c: int, q: int, kind: str, theta: Optional[float] = None):
        self.ensure_col(c)
        # remove any ops using this qubit in this column
        self.remove_ops_touching(c, [q])
        op = Op(kind=kind, q=q)
        if kind in ("RX","RY","RZ"):
            op.theta = float(theta if theta is not None else 0.0)
        self.columns[c].append(op)

    def place_span(self, c: int, kind: str):
        self.ensure_col(c)
        self.wipe_col(c)
        self.columns[c].append(Op(kind=kind, q=None))

    def place_cnot(self, c: int, control: int, target: int):
        self.ensure_col(c)
        self.remove_ops_touching(c, [control, target])
        self.columns[c].append(Op(kind="CNOT", control=control, target=target))

    def place_cz(self, c: int, control: int, target: int):
        self.ensure_col(c)
        self.remove_ops_touching(c, [control, target])
        self.columns[c].append(Op(kind="CZ", control=control, target=target))

    def place_swap(self, c: int, q1: int, q2: int):
        self.ensure_col(c)
        self.remove_ops_touching(c, [q1, q2])
        self.columns[c].append(Op(kind="SWAP", q=q1, q2=q2))

    def delete_at(self, c: int, q: int):
        if not (0 <= c < len(self.columns)):
            return
        new_ops = []
        removed_any = False
        for op in self.columns[c]:
            if op.kind in SPAN_KINDS:
                # delete span if cursor hits any row in that column
                removed_any = True
                continue
            if q in op.touches():
                removed_any = True
                continue
            new_ops.append(op)
        if removed_any:
            self.columns[c] = new_ops

    def add_qubit(self):
        self.n += 1

    def remove_last_qubit(self):
        if self.n <= 1:
            return
        q_del = self.n - 1
        # remove ops touching last qubit
        for c in range(len(self.columns)):
            new_ops = []
            for op in self.columns[c]:
                if op.kind in SPAN_KINDS:
                    new_ops.append(op)
                    continue
                if q_del in op.touches():
                    continue
                new_ops.append(op)
            self.columns[c] = new_ops
        self.n -= 1


# =========================
# Simulation
# =========================
def apply_column(psi: np.ndarray, circ: Circuit, c: int) -> np.ndarray:
    if not (0 <= c < len(circ.columns)):
        return psi
    ops = circ.columns[c]

    # span ops handled once
    has_oracle = any(op.kind == "ORACLE" for op in ops)
    has_diff = any(op.kind == "DIFF" for op in ops)

    out = psi

    # Apply 1q and 2q ops (order: 1q first then 2q, mostly fine for demos)
    for op in ops:
        if op.kind in SPAN_KINDS:
            continue
        if op.kind == "H":
            out = apply_1q(out, circ.n, op.q, gate_H())
        elif op.kind == "X":
            out = apply_1q(out, circ.n, op.q, gate_X())
        elif op.kind == "Y":
            out = apply_1q(out, circ.n, op.q, gate_Y())
        elif op.kind == "Z":
            out = apply_1q(out, circ.n, op.q, gate_Z())
        elif op.kind == "S":
            out = apply_1q(out, circ.n, op.q, gate_S())
        elif op.kind == "T":
            out = apply_1q(out, circ.n, op.q, gate_T())
        elif op.kind == "RX":
            out = apply_1q(out, circ.n, op.q, gate_RX(op.theta or 0.0))
        elif op.kind == "RY":
            out = apply_1q(out, circ.n, op.q, gate_RY(op.theta or 0.0))
        elif op.kind == "RZ":
            out = apply_1q(out, circ.n, op.q, gate_RZ(op.theta or 0.0))
        elif op.kind == "CNOT":
            out = apply_cnot(out, circ.n, op.control, op.target)
        elif op.kind == "CZ":
            out = apply_cz(out, circ.n, op.control, op.target)
        elif op.kind == "SWAP":
            out = apply_swap(out, circ.n, op.q, op.q2)
        elif op.kind == "MEAS":
            # no-op in statevector (measurement handled via shots)
            pass

    if has_oracle:
        # Grover oracle: phase flip marked basis state
        out = out.copy()
        m = int(clamp(circ.oracle_mark, 0, (1 << circ.n) - 1))
        out[m] *= -1.0

    if has_diff:
        # Grover diffusion: psi' = 2*avg - psi
        N = out.shape[0]
        avg = np.sum(out) / float(N)
        out = (2.0 * avg) - out

    return out

def state_after_playhead(circ: Circuit, play_col: int) -> np.ndarray:
    dim = 1 << circ.n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0j
    if play_col < 0:
        return psi
    for c in range(min(play_col + 1, len(circ.columns))):
        psi = apply_column(psi, circ, c)
    return psi


# =========================
# Shots rolling window
# =========================
@dataclass
class ShotWindow:
    dim: int
    window: int
    ring: np.ndarray
    pos: int
    filled: int
    counts: np.ndarray

    @classmethod
    def create(cls, dim: int, window: int):
        ring = np.full(window, -1, dtype=np.int32)
        counts = np.zeros(dim, dtype=np.int32)
        return cls(dim, window, ring, 0, 0, counts)

    def clear(self):
        self.ring.fill(-1)
        self.counts.fill(0)
        self.pos = 0
        self.filled = 0

    def push(self, outcomes: np.ndarray):
        # outcomes: 1D array
        for o in outcomes:
            o = int(o)
            if self.filled < self.window:
                self.ring[self.pos] = o
                self.counts[o] += 1
                self.filled += 1
            else:
                old = int(self.ring[self.pos])
                if old >= 0:
                    self.counts[old] -= 1
                self.ring[self.pos] = o
                self.counts[o] += 1
            self.pos += 1
            if self.pos >= self.window:
                self.pos = 0

    def freqs(self) -> np.ndarray:
        if self.filled <= 0:
            return np.zeros(self.dim, dtype=np.float64)
        return self.counts.astype(np.float64) / float(self.filled)


# =========================
# Drawing primitives
# =========================
SHADES = " .:-=+*#%@"
def shade_char(t01: float) -> str:
    t = clamp(t01, 0.0, 1.0)
    idx = int(t * (len(SHADES) - 1) + 1e-9)
    return SHADES[idx]

def draw_bar(stdscr, y, x, w, frac, label, attr=0):
    frac = clamp(frac, 0.0, 1.0)
    fill = int(frac * w)
    safe_addstr(stdscr, y, x, f"{label} ", curses.A_BOLD)
    x2 = x + len(label) + 1
    safe_addstr(stdscr, y, x2, "█" * fill, attr)
    safe_addstr(stdscr, y, x2 + fill, " " * (w - fill))
    safe_addstr(stdscr, y, x2 + w + 1, f"{frac:6.3f}")

def draw_phase_legend(stdscr, y, x, phase_attr, bins=12):
    safe_addstr(stdscr, y, x, "phase:", curses.A_BOLD)
    for k in range(bins):
        ang = -math.pi + (2 * math.pi) * (k / bins)
        safe_addstr(stdscr, y, x + 7 + k, "█", phase_attr(ang))
    safe_addstr(stdscr, y, x + 7 + bins + 1, "-π", curses.A_DIM)
    safe_addstr(stdscr, y, x + 7 + bins + 5, "0", curses.A_DIM)
    safe_addstr(stdscr, y, x + 7 + bins + 8, "+π", curses.A_DIM)

def draw_bloch_2d(stdscr, y0, x0, w, h, plane, bx, by, bz):
    if w < 13 or h < 9:
        safe_addstr(stdscr, y0, x0, "Bloch (resize)", curses.A_DIM)
        return
    safe_addstr(stdscr, y0, x0, f"Bloch 2D ({plane})", curses.A_BOLD)
    yy0 = y0 + 1
    xx0 = x0
    hh = h - 1
    ww = w
    cx = xx0 + ww // 2
    cy = yy0 + hh // 2
    r = min(ww, hh) // 2 - 1
    if r < 3:
        return

    if plane == "XY":
        px, py = bx, by
        xlab, ylab = "X", "Y"
    else:
        px, py = bx, bz
        xlab, ylab = "X", "Z"

    for yy in range(yy0, yy0 + hh):
        for xx in range(xx0, xx0 + ww):
            dx = (xx - cx)
            dy = (yy - cy)
            dist = math.sqrt(dx * dx + dy * dy)
            if abs(dist - r) < 0.55:
                safe_addstr(stdscr, yy, xx, "·", curses.A_DIM)

    safe_addstr(stdscr, cy, cx + r + 1, f"+{xlab}", curses.A_DIM)
    safe_addstr(stdscr, cy, cx - r - 3, f"-{xlab}", curses.A_DIM)
    safe_addstr(stdscr, cy - r - 1, cx - 1, f"+{ylab}", curses.A_DIM)
    safe_addstr(stdscr, cy + r + 1, cx - 1, f"-{ylab}", curses.A_DIM)

    tx = cx + int(px * r)
    ty = cy - int(py * r)
    steps = max(abs(tx - cx), abs(ty - cy), 1)
    for i in range(steps):
        fx = cx + (tx - cx) * (i / steps)
        fy = cy + (ty - cy) * (i / steps)
        safe_addstr(stdscr, int(round(fy)), int(round(fx)), "*")

    dx = tx - cx
    dy = ty - cy
    if abs(dx) > abs(dy):
        head = ">" if dx > 0 else "<"
    else:
        head = "^" if dy < 0 else "v"
    safe_addstr(stdscr, ty, tx, head, curses.A_BOLD)

def draw_amp_list(stdscr, y0, x0, w, h, psi, n, phase_attr, topk=16, sort_by_prob=True):
    p = probs_from_state(psi)
    mags = np.sqrt(p)
    phases = np.angle(psi)

    idxs = np.arange(1 << n)
    if sort_by_prob:
        idxs = idxs[np.argsort(-p)]
    if topk > 0:
        idxs = idxs[:min(topk, len(idxs))]

    safe_addstr(stdscr, y0, x0, "Amplitudes (|amp| + phase)", curses.A_BOLD)
    y = y0 + 1
    safe_addstr(stdscr, y, x0, "basis".ljust(n + 3) + " |amp|   prob     phase      bar", curses.A_UNDERLINE)
    y += 1

    bar_x = x0 + min(w - 10, (n + 3) + len(" |amp|   prob     phase      "))
    bar_w = max(6, (x0 + w - 1) - bar_x)

    max_mag = float(mags[idxs].max()) if idxs.size else 1.0
    if max_mag <= 0:
        max_mag = 1.0

    for i in idxs:
        if y >= y0 + h:
            break
        mag = float(mags[i])
        prob = float(p[i])
        ph = float(phases[i])
        label = basis_label(int(i), n)[: max(0, n + 3)]
        row = f"{label:<{n+3}} {mag:6.4f} {prob:7.4f} {fmt_deg(ph):>9}  "
        safe_addstr(stdscr, y, x0, row[:w-1])
        frac = clamp(mag / max_mag, 0.0, 1.0)
        fill = int(frac * bar_w)
        safe_addstr(stdscr, y, bar_x, "█" * fill, phase_attr(ph))
        y += 1

def draw_corr_heatmap(stdscr, y0, x0, w, h, M, heat_attr, title):
    n = M.shape[0]
    safe_addstr(stdscr, y0, x0, title[:w-1], curses.A_BOLD)
    y = y0 + 1
    if y >= y0 + h:
        return
    cell_w = 2
    max_cols = max(1, (w - 6) // cell_w)
    show_n = min(n, max_cols)
    safe_addstr(stdscr, y, x0 + 4, " ".join([f"{i:d}" for i in range(show_n)])[:w-1], curses.A_DIM)
    y += 1
    for i in range(show_n):
        if y >= y0 + h:
            break
        safe_addstr(stdscr, y, x0, f"q{i}: ", curses.A_DIM)
        x = x0 + 4
        for j in range(show_n):
            t = float(M[i, j])
            safe_addstr(stdscr, y, x, shade_char(t), heat_attr(t))
            safe_addstr(stdscr, y, x + 1, " ")
            x += 2
        y += 1


# =========================
# Circuit grid rendering
# =========================
def gate_cell(label: str) -> str:
    # 5 chars, wire-ish
    if label == "H":
        return "─[H]─"
    if label == "X":
        return "─[X]─"
    if label == "Y":
        return "─[Y]─"
    if label == "Z":
        return "─[Z]─"
    if label == "S":
        return "─[S]─"
    if label == "T":
        return "─[T]─"
    if label == "Rx":
        return "─Rx─ "
    if label == "Ry":
        return "─Ry─ "
    if label == "Rz":
        return "─Rz─ "
    if label == "M":
        return "─[M]─"
    if label == "O":
        return "─[O]─"
    if label == "D":
        return "─[D]─"
    return (f"[{label}]")[:5].ljust(5)

def draw_circuit(
    stdscr,
    y0: int,
    x0: int,
    w: int,
    circ: Circuit,
    play_col: int,
    edit_row: int,
    edit_col: int,
    scroll_col: int,
    focus: str,
    pending: Optional[Tuple[str,int,int]],  # (kind, col, first_row)
    c_hdr=0,
):
    n = circ.n
    cols = len(circ.columns)

    label_w = 4  # "q0:"
    cell_w = 6   # 5 + space
    content_w = max(0, w - label_w - 1)
    vis_cols = max(1, content_w // cell_w)

    start = int(clamp(scroll_col, 0, max(0, cols - vis_cols)))
    end = min(cols, start + vis_cols)

    # Title row
    title = "Circuit builder (EDIT cursor + PLAY head)"
    safe_addstr(stdscr, y0, x0, title[:w-1], curses.A_BOLD)

    # Column index row
    y = y0 + 1
    safe_addstr(stdscr, y, x0, " " * label_w)
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        attr = curses.A_REVERSE if ci == play_col else curses.A_DIM
        safe_addstr(stdscr, y, x, f"{ci:>2}", attr)

    # Param hint row (show one param gate value if any)
    y = y0 + 2
    safe_addstr(stdscr, y, x0, " " * label_w)
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        # find first param gate
        th = None
        for op in circ.columns[ci]:
            if op.kind in ("RX","RY","RZ"):
                th = op.theta
                break
        if th is not None:
            s = f"{th:+.2f}"[:5].ljust(5)
            attr = curses.A_REVERSE if ci == play_col else curses.A_DIM
            safe_addstr(stdscr, y, x, s, attr)

    # Rows baseline wires
    y_rows = y0 + 3
    for q in range(n):
        safe_addstr(stdscr, y_rows + q, x0, f"q{q}:".ljust(label_w), curses.A_BOLD | c_hdr)
        for ci in range(start, end):
            x = x0 + label_w + (ci - start) * cell_w
            attr = 0
            if ci == play_col:
                attr |= curses.A_REVERSE
            # if edit cell in this row+col, standout
            if (q == edit_row) and (ci == edit_col):
                attr |= curses.A_STANDOUT
            safe_addstr(stdscr, y_rows + q, x, "─────", attr)

    # Overlay span ops (Oracle/Diff)
    for ci in range(start, end):
        span_kind = None
        for op in circ.columns[ci]:
            if op.kind in SPAN_KINDS:
                span_kind = op.kind
                break
        if span_kind:
            x = x0 + label_w + (ci - start) * cell_w
            col_attr = curses.A_REVERSE if ci == play_col else 0
            lab = "O" if span_kind == "ORACLE" else "D"
            for q in range(n):
                attr = col_attr
                if (q == edit_row) and (ci == edit_col):
                    attr |= curses.A_STANDOUT
                safe_addstr(stdscr, y_rows + q, x, gate_cell(lab), attr)

    # Overlay CNOT/CZ/SWAP (vertical)
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        col_attr = curses.A_REVERSE if ci == play_col else 0

        for op in circ.columns[ci]:
            if op.kind == "CNOT":
                c = op.control; t = op.target
                y_c = y_rows + c
                y_t = y_rows + t
                y_min, y_max = (y_c, y_t) if y_c <= y_t else (y_t, y_c)
                for yy in range(y_min + 1, y_max):
                    safe_addstr(stdscr, yy, x + 2, "│", col_attr)
                safe_addstr(stdscr, y_c, x + 2, "●", col_attr | curses.A_BOLD)
                safe_addstr(stdscr, y_t, x + 2, "X", col_attr | curses.A_BOLD)
            elif op.kind == "CZ":
                c = op.control; t = op.target
                y_c = y_rows + c
                y_t = y_rows + t
                y_min, y_max = (y_c, y_t) if y_c <= y_t else (y_t, y_c)
                for yy in range(y_min + 1, y_max):
                    safe_addstr(stdscr, yy, x + 2, "│", col_attr)
                safe_addstr(stdscr, y_c, x + 2, "●", col_attr | curses.A_BOLD)
                safe_addstr(stdscr, y_t, x + 2, "Z", col_attr | curses.A_BOLD)
            elif op.kind == "SWAP":
                q1 = op.q; q2 = op.q2
                y_1 = y_rows + q1
                y_2 = y_rows + q2
                y_min, y_max = (y_1, y_2) if y_1 <= y_2 else (y_2, y_1)
                for yy in range(y_min + 1, y_max):
                    safe_addstr(stdscr, yy, x + 2, "│", col_attr)
                safe_addstr(stdscr, y_1, x + 2, "x", col_attr | curses.A_BOLD)
                safe_addstr(stdscr, y_2, x + 2, "x", col_attr | curses.A_BOLD)

    # Overlay 1q gates
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        col_attr = curses.A_REVERSE if ci == play_col else 0
        for op in circ.columns[ci]:
            if op.kind in ONEQ_KINDS:
                q = op.q
                if q is None: 
                    continue
                lab = None
                if op.kind in ("H","X","Y","Z","S","T"):
                    lab = op.kind
                elif op.kind == "RX":
                    lab = "Rx"
                elif op.kind == "RY":
                    lab = "Ry"
                elif op.kind == "RZ":
                    lab = "Rz"
                elif op.kind == "MEAS":
                    lab = "M"
                if lab is None:
                    continue
                attr = col_attr
                if (q == edit_row) and (ci == edit_col):
                    attr |= curses.A_STANDOUT
                safe_addstr(stdscr, y_rows + q, x, gate_cell(lab), attr)

    # Pending placement marker
    if pending is not None:
        pk, pc, pr = pending
        if start <= pc < end:
            x = x0 + label_w + (pc - start) * cell_w
            y = y_rows + pr
            safe_addstr(stdscr, y, x, gate_cell("?"), curses.A_BOLD | curses.A_DIM)

    # Footer
    footer_y = y_rows + n
    focus_txt = "EDIT" if focus == "edit" else "PLAY"
    safe_addstr(stdscr, footer_y, x0, f"focus={focus_txt}  cols={cols}  view={start}..{end-1}  scroll < >"[:w-1], curses.A_DIM)
    return footer_y + 1


# =========================
# Prompts
# =========================
def prompt_line(stdscr, prompt: str, default: str = "") -> Optional[str]:
    h, w = stdscr.getmaxyx()
    y = h - 2
    stdscr.nodelay(False)
    curses.echo()
    try:
        safe_addstr(stdscr, y, 0, " " * (w - 1))
        safe_addstr(stdscr, y, 0, prompt[:w-1], curses.A_BOLD)
        if default:
            safe_addstr(stdscr, y, min(w - 1, len(prompt)), default[: max(0, w - 1 - len(prompt))])
        stdscr.move(y, min(w - 1, len(prompt) + len(default)))
        s = stdscr.getstr(y, min(w - 1, len(prompt) + len(default)), 120)
        if s is None:
            return None
        txt = s.decode("utf-8", errors="ignore").strip()
        if not txt and default:
            txt = default
        return txt
    finally:
        curses.noecho()
        stdscr.nodelay(True)


# =========================
# Templates
# =========================
def template_empty(n: int = 2, cols: int = 8) -> Circuit:
    return Circuit(n=n, columns=[[] for _ in range(cols)], oracle_mark=0)

def template_interference() -> Circuit:
    c = Circuit(n=1, columns=[[] for _ in range(3)], oracle_mark=0)
    c.place_oneq(0, 0, "H")
    c.place_oneq(1, 0, "RZ", theta=0.0)
    c.place_oneq(2, 0, "H")
    return c

def template_bell() -> Circuit:
    c = Circuit(n=2, columns=[[] for _ in range(4)], oracle_mark=0)
    c.place_oneq(0, 0, "H")
    c.place_cnot(1, 0, 1)
    c.place_oneq(2, 0, "RZ", theta=0.0)
    # leave col3 empty for builder expansion
    return c

def template_grover3(mark: int = 5, iters: int = 1) -> Circuit:
    cols = []
    # H on all
    col0 = [Op("H", q=0), Op("H", q=1), Op("H", q=2)]
    cols.append(col0)
    for _ in range(iters):
        cols.append([Op("ORACLE")])
        cols.append([Op("DIFF")])
    c = Circuit(n=3, columns=cols, oracle_mark=mark)
    return c


# =========================
# Undo/Redo snapshots
# =========================
@dataclass
class Snapshot:
    circuit: Dict
    play_col: int
    edit_row: int
    edit_col: int
    scroll_col: int
    selected_q: int
    metric_mode: str
    plane: str

def make_snapshot(circ: Circuit, play_col: int, edit_row: int, edit_col: int, scroll_col: int,
                  selected_q: int, metric_mode: str, plane: str) -> Snapshot:
    return Snapshot(
        circuit=circ.to_dict(),
        play_col=play_col,
        edit_row=edit_row,
        edit_col=edit_col,
        scroll_col=scroll_col,
        selected_q=selected_q,
        metric_mode=metric_mode,
        plane=plane,
    )

def apply_snapshot(s: Snapshot):
    return (Circuit.from_dict(s.circuit), s.play_col, s.edit_row, s.edit_col, s.scroll_col,
            s.selected_q, s.metric_mode, s.plane)


# =========================
# Main App
# =========================
HELP_LINES = [
"FOCUS: TAB toggles EDIT/PLAY.",
"PLAY focus: ←/→ moves play head (applied column). a toggles autoplay, +/- speed, r reset play head.",
"EDIT focus: arrows move cell; place gates: h x y z s t  (H/X/Y/Z/S/T),  r/e/w (Rz/Ry/Rx), m (measure).",
"2-qubit: c (CNOT), v (CZ), p (SWAP): press once to pick first qubit, move row, press again to place.",
"Columns: I insert before, D delete column, A append new column.",
"Q add qubit, K remove last qubit.",
"Param edit: [ ] adjust θ of Rx/Ry/Rz at cursor, ENTER set exact θ.",
"Clipboard: Y yank column, P paste before, p (in EDIT) is SWAP; so use V to paste after (see below).",
"Undo/Redo: u / U.",
"Save/Load: S save JSON, L load JSON, N new empty.",
"Panels: k cycle Bloch qubit, P (PLAY) toggles plane XY/XZ, M (PLAY) toggles metric corr/MI, X toggles shots.",
"Templates: 1 empty(2q), 2 interference, 3 bell, 4 grover3.",
"? toggles this help.",
]

@dataclass
class AppState:
    focus: str = "edit"          # "edit" or "play"
    running: bool = True
    autoplay: bool = False
    col_rate: float = 2.0

    play_col: int = 0
    edit_row: int = 0
    edit_col: int = 0
    scroll_col: int = 0

    selected_q: int = 0
    plane: str = "XY"
    metric_mode: str = "corr"    # corr or mi

    topk: int = 16
    sort_by_prob: bool = True

    show_shots: bool = False
    shots_per_tick: int = 400
    shots_window: int = 6000

    pending: Optional[Tuple[str,int,int]] = None  # (kind, col, first_row)
    clipboard_col: Optional[List[Op]] = None

    show_help: bool = False
    status_msg: str = ""
    status_ttl: float = 0.0


def set_status(app: AppState, msg: str, ttl: float = 2.0):
    app.status_msg = msg
    app.status_ttl = ttl


def adjust_scroll_to_include(app: AppState, cols_visible: int, target_col: int, total_cols: int):
    if total_cols <= cols_visible:
        app.scroll_col = 0
        return
    start = app.scroll_col
    end = start + cols_visible - 1
    if target_col < start:
        app.scroll_col = target_col
    elif target_col > end:
        app.scroll_col = target_col - (cols_visible - 1)
    app.scroll_col = int(clamp(app.scroll_col, 0, max(0, total_cols - cols_visible)))


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    locale.setlocale(locale.LC_ALL, "")

    phase_attr, heat_attr, c_hdr, c_met, c_prob, c_shot, c_err = init_palettes(stdscr)

    circ = template_bell()
    app = AppState()
    app.play_col = 0
    app.edit_col = 0
    app.edit_row = 0
    app.selected_q = 0

    undo: List[Snapshot] = []
    redo: List[Snapshot] = []

    shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
    rng = np.random.default_rng(1)

    acc = 0.0
    last = time.time()
    t0 = last

    def push_undo():
        undo.append(make_snapshot(circ, app.play_col, app.edit_row, app.edit_col, app.scroll_col,
                                  app.selected_q, app.metric_mode, app.plane))
        if len(undo) > 200:
            undo.pop(0)
        redo.clear()

    def cancel_pending():
        app.pending = None

    while True:
        now = time.time()
        dt = now - last
        last = now
        t = now - t0

        if app.status_ttl > 0:
            app.status_ttl -= dt
            if app.status_ttl <= 0:
                app.status_msg = ""

        # Read input
        ch = -1
        try:
            ch = stdscr.getch()
        except curses.error:
            pass

        semantic_changed = False  # clear shots
        circuit_changed = False

        if ch != -1:
            # Global quit / help
            if ch in (ord('q'), ord('Q')):
                break
            if ch == ord('?'):
                app.show_help = not app.show_help

            # Focus toggle
            if ch == 9:  # TAB
                app.focus = "play" if app.focus == "edit" else "edit"
                cancel_pending()
                set_status(app, f"focus -> {app.focus.upper()}")
            # Templates (global)
            if ch in (ord('1'), ord('2'), ord('3'), ord('4')):
                push_undo()
                if ch == ord('1'):
                    circ = template_empty(2, 8)
                    set_status(app, "template: empty 2q")
                elif ch == ord('2'):
                    circ = template_interference()
                    set_status(app, "template: interference")
                elif ch == ord('3'):
                    circ = template_bell()
                    set_status(app, "template: bell")
                elif ch == ord('4'):
                    circ = template_grover3(mark=5, iters=1)
                    set_status(app, "template: grover n=3")
                app.play_col = 0
                app.edit_col = 0
                app.edit_row = 0
                app.selected_q = 0
                app.scroll_col = 0
                shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                semantic_changed = True
                circuit_changed = True
                cancel_pending()

            # Undo/redo
            if ch == ord('u'):
                if undo:
                    redo.append(make_snapshot(circ, app.play_col, app.edit_row, app.edit_col, app.scroll_col,
                                              app.selected_q, app.metric_mode, app.plane))
                    s = undo.pop()
                    circ, app.play_col, app.edit_row, app.edit_col, app.scroll_col, app.selected_q, app.metric_mode, app.plane = apply_snapshot(s)
                    shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                    semantic_changed = True
                    cancel_pending()
                    set_status(app, "undo")
            if ch == ord('U'):
                if redo:
                    undo.append(make_snapshot(circ, app.play_col, app.edit_row, app.edit_col, app.scroll_col,
                                              app.selected_q, app.metric_mode, app.plane))
                    s = redo.pop()
                    circ, app.play_col, app.edit_row, app.edit_col, app.scroll_col, app.selected_q, app.metric_mode, app.plane = apply_snapshot(s)
                    shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                    semantic_changed = True
                    cancel_pending()
                    set_status(app, "redo")

            # Save/load/new
            if ch == ord('S'):
                name = prompt_line(stdscr, "Save JSON filename: ", "circuit.json")
                if name:
                    try:
                        with open(name, "w", encoding="utf-8") as f:
                            json.dump(circ.to_dict(), f, indent=2)
                        set_status(app, f"saved: {name}")
                    except Exception as e:
                        set_status(app, f"save error: {e}", ttl=3.0)
            if ch == ord('L'):
                name = prompt_line(stdscr, "Load JSON filename: ", "circuit.json")
                if name:
                    try:
                        with open(name, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        push_undo()
                        circ = Circuit.from_dict(d)
                        app.play_col = int(clamp(app.play_col, 0, len(circ.columns) - 1))
                        app.edit_col = int(clamp(app.edit_col, 0, len(circ.columns) - 1))
                        app.edit_row = int(clamp(app.edit_row, 0, circ.n - 1))
                        app.selected_q = int(clamp(app.selected_q, 0, circ.n - 1))
                        shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                        semantic_changed = True
                        cancel_pending()
                        set_status(app, f"loaded: {name}")
                    except Exception as e:
                        set_status(app, f"load error: {e}", ttl=3.0)
            if ch == ord('N'):
                push_undo()
                circ = template_empty(2, 8)
                app.play_col = 0
                app.edit_col = 0
                app.edit_row = 0
                app.selected_q = 0
                app.scroll_col = 0
                shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                semantic_changed = True
                cancel_pending()
                set_status(app, "new empty circuit")

            # Global panel toggles (in PLAY focus or uppercase)
            if (app.focus == "play") or (ch in (ord('P'), ord('M'), ord('X'))):
                if ch in (ord('k'), ord('K')) and app.focus == "play":
                    app.selected_q = (app.selected_q + 1) % circ.n
                if ch == ord('P'):  # plane toggle
                    app.plane = "XZ" if app.plane == "XY" else "XY"
                if ch == ord('M'):  # metric toggle
                    app.metric_mode = "mi" if app.metric_mode == "corr" else "corr"
                if ch == ord('X'):  # shots toggle
                    app.show_shots = not app.show_shots
                    shotwin.clear()
                if ch in (ord('t'), ord('T')) and app.focus == "play":
                    # cycle topK 8/16/32/all
                    if app.topk == 8:
                        app.topk = 16
                    elif app.topk == 16:
                        app.topk = 32
                    elif app.topk == 32:
                        app.topk = 0
                    else:
                        app.topk = 8
                if ch in (ord('s'), ord('S')) and app.focus == "play":
                    app.sort_by_prob = not app.sort_by_prob

            # Focus-specific controls
            if app.focus == "play":
                if ch == ord(' '):
                    app.running = not app.running
                elif ch in (ord('a'), ord('A')):
                    app.autoplay = not app.autoplay
                    set_status(app, f"autoplay {'ON' if app.autoplay else 'OFF'}")
                elif ch in (ord('+'), ord('=')):
                    app.col_rate = min(20.0, app.col_rate * 1.25)
                elif ch in (ord('-'), ord('_')):
                    app.col_rate = max(0.2, app.col_rate / 1.25)
                elif ch in (curses.KEY_LEFT, ord('h')):
                    app.autoplay = False
                    app.play_col = max(0, app.play_col - 1)
                elif ch in (curses.KEY_RIGHT, ord('l')):
                    app.autoplay = False
                    app.play_col = min(len(circ.columns) - 1, app.play_col + 1)
                elif ch == ord('r'):
                    app.play_col = 0
                elif ch == ord('<'):
                    app.scroll_col = max(0, app.scroll_col - 1)
                elif ch == ord('>'):
                    app.scroll_col = min(max(0, len(circ.columns) - 1), app.scroll_col + 1)

            else:  # EDIT focus
                # navigation
                if ch == curses.KEY_UP:
                    app.edit_row = max(0, app.edit_row - 1)
                    app.selected_q = app.edit_row
                elif ch == curses.KEY_DOWN:
                    app.edit_row = min(circ.n - 1, app.edit_row + 1)
                    app.selected_q = app.edit_row
                elif ch == curses.KEY_LEFT:
                    app.edit_col = max(0, app.edit_col - 1)
                    cancel_pending()
                elif ch == curses.KEY_RIGHT:
                    # auto-extend circuit when moving beyond end
                    if app.edit_col >= len(circ.columns) - 1:
                        push_undo()
                        circ.insert_col(len(circ.columns))
                        circuit_changed = True
                        semantic_changed = True
                    app.edit_col = min(len(circ.columns) - 1, app.edit_col + 1)
                    cancel_pending()
                elif ch == ord('<'):
                    app.scroll_col = max(0, app.scroll_col - 1)
                elif ch == ord('>'):
                    app.scroll_col = min(max(0, len(circ.columns) - 1), app.scroll_col + 1)

                # column ops
                if ch == ord('I'):
                    push_undo()
                    circ.insert_col(app.edit_col)
                    circuit_changed = True
                    semantic_changed = True
                    cancel_pending()
                elif ch == ord('D'):
                    push_undo()
                    circ.delete_col(app.edit_col)
                    app.edit_col = int(clamp(app.edit_col, 0, len(circ.columns) - 1))
                    app.play_col = int(clamp(app.play_col, 0, len(circ.columns) - 1))
                    circuit_changed = True
                    semantic_changed = True
                    cancel_pending()
                elif ch == ord('A'):
                    push_undo()
                    circ.insert_col(len(circ.columns))
                    app.edit_col = len(circ.columns) - 1
                    circuit_changed = True
                    semantic_changed = True
                    cancel_pending()

                # qubit ops
                if ch == ord('Q'):
                    push_undo()
                    circ.add_qubit()
                    app.edit_row = int(clamp(app.edit_row, 0, circ.n - 1))
                    app.selected_q = app.edit_row
                    shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                    semantic_changed = True
                    circuit_changed = True
                    cancel_pending()
                elif ch == ord('K'):
                    push_undo()
                    circ.remove_last_qubit()
                    app.edit_row = int(clamp(app.edit_row, 0, circ.n - 1))
                    app.selected_q = app.edit_row
                    shotwin = ShotWindow.create(dim=1 << circ.n, window=app.shots_window)
                    semantic_changed = True
                    circuit_changed = True
                    cancel_pending()

                # clipboard (column)
                if ch == ord('Y'):
                    if 0 <= app.edit_col < len(circ.columns):
                        app.clipboard_col = copy.deepcopy(circ.columns[app.edit_col])
                        set_status(app, "yanked column")
                elif ch == ord('V'):  # paste AFTER
                    if app.clipboard_col is not None:
                        push_undo()
                        circ.insert_col(app.edit_col + 1)
                        circ.columns[app.edit_col + 1] = copy.deepcopy(app.clipboard_col)
                        app.edit_col += 1
                        semantic_changed = True
                        circuit_changed = True
                        set_status(app, "pasted after")
                elif ch == ord('P'):  # paste BEFORE
                    if app.clipboard_col is not None:
                        push_undo()
                        circ.insert_col(app.edit_col)
                        circ.columns[app.edit_col] = copy.deepcopy(app.clipboard_col)
                        semantic_changed = True
                        circuit_changed = True
                        set_status(app, "pasted before")

                # delete gate at cursor
                if ch in (ord('d'), curses.KEY_BACKSPACE, 127, 8):
                    push_undo()
                    circ.delete_at(app.edit_col, app.edit_row)
                    semantic_changed = True
                    circuit_changed = True
                    cancel_pending()

                # span ops (Grover)
                if ch == ord('O'):
                    push_undo()
                    circ.place_span(app.edit_col, "ORACLE")
                    semantic_changed = True
                    circuit_changed = True
                    set_status(app, "placed ORACLE (span)")
                if ch == ord('F'):
                    push_undo()
                    circ.place_span(app.edit_col, "DIFF")
                    semantic_changed = True
                    circuit_changed = True
                    set_status(app, "placed DIFF (span)")
                if ch == ord('G'):
                    # set oracle mark
                    s = prompt_line(stdscr, "Grover mark (0..2^n-1): ", str(circ.oracle_mark))
                    if s is not None:
                        try:
                            v = int(s, 0)
                            push_undo()
                            circ.oracle_mark = int(clamp(v, 0, (1 << circ.n) - 1))
                            semantic_changed = True
                            set_status(app, f"oracle_mark={circ.oracle_mark}")
                        except Exception:
                            set_status(app, "invalid mark", ttl=2.0)

                # 1q placement keys
                def place_1q(kind: str, theta: Optional[float] = None):
                    nonlocal semantic_changed, circuit_changed
                    push_undo()
                    circ.place_oneq(app.edit_col, app.edit_row, kind, theta=theta)
                    semantic_changed = True
                    circuit_changed = True
                    cancel_pending()

                if ch in (ord('h'), ord('x'), ord('y'), ord('z'), ord('s'), ord('t')):
                    mapping = {
                        ord('h'): "H", ord('x'): "X", ord('y'): "Y", ord('z'): "Z",
                        ord('s'): "S", ord('t'): "T",
                    }
                    place_1q(mapping[ch])
                elif ch == ord('m'):
                    place_1q("MEAS")
                elif ch == ord('r'):
                    place_1q("RZ", theta=0.0)
                elif ch == ord('e'):
                    place_1q("RY", theta=0.0)
                elif ch == ord('w'):
                    place_1q("RX", theta=0.0)

                # multi-qubit placement: two-tap
                def begin_or_finish(kind: str):
                    nonlocal semantic_changed, circuit_changed
                    if app.pending is None:
                        app.pending = (kind, app.edit_col, app.edit_row)
                        set_status(app, f"{kind}: select other qubit then press again")
                        return
                    pk, pc, pr = app.pending
                    if pk != kind or pc != app.edit_col:
                        # restart in this column
                        app.pending = (kind, app.edit_col, app.edit_row)
                        set_status(app, f"{kind}: select other qubit then press again")
                        return
                    # finish
                    q1 = pr
                    q2 = app.edit_row
                    if q1 == q2:
                        set_status(app, "pick a different qubit", ttl=2.0)
                        return
                    push_undo()
                    if kind == "CNOT":
                        circ.place_cnot(app.edit_col, control=q1, target=q2)
                    elif kind == "CZ":
                        circ.place_cz(app.edit_col, control=q1, target=q2)
                    elif kind == "SWAP":
                        circ.place_swap(app.edit_col, q1=q1, q2=q2)
                    semantic_changed = True
                    circuit_changed = True
                    app.pending = None
                    set_status(app, f"placed {kind}")

                if ch == ord('c'):
                    begin_or_finish("CNOT")
                elif ch == ord('v'):
                    begin_or_finish("CZ")
                elif ch == ord('p'):
                    begin_or_finish("SWAP")

                # parameter adjust
                if ch in (ord('['), ord(']'), 10, 13):  # enter
                    # find param gate at cursor
                    col = circ.columns[app.edit_col] if 0 <= app.edit_col < len(circ.columns) else []
                    param_op = None
                    for op in col:
                        if op.kind in ("RX","RY","RZ") and op.q == app.edit_row:
                            param_op = op
                            break
                    if param_op is not None:
                        if ch == ord('['):
                            push_undo()
                            param_op.theta = wrap_pi((param_op.theta or 0.0) - 0.10)
                            semantic_changed = True
                            circuit_changed = True
                            set_status(app, f"θ={param_op.theta:+.2f}")
                        elif ch == ord(']'):
                            push_undo()
                            param_op.theta = wrap_pi((param_op.theta or 0.0) + 0.10)
                            semantic_changed = True
                            circuit_changed = True
                            set_status(app, f"θ={param_op.theta:+.2f}")
                        else:
                            s = prompt_line(stdscr, "Set θ (radians): ", f"{param_op.theta or 0.0:+.6f}")
                            if s is not None:
                                try:
                                    v = float(s)
                                    push_undo()
                                    param_op.theta = wrap_pi(v)
                                    semantic_changed = True
                                    circuit_changed = True
                                    set_status(app, f"θ={param_op.theta:+.3f}")
                                except Exception:
                                    set_status(app, "invalid θ", ttl=2.0)

        # Autoplay play head
        if app.running and app.autoplay:
            acc += dt * app.col_rate
            while acc >= 1.0:
                acc -= 1.0
                app.play_col += 1
                if app.play_col >= len(circ.columns):
                    app.play_col = 0

        # If circuit changed, keep play/edit within bounds
        app.play_col = int(clamp(app.play_col, 0, len(circ.columns) - 1))
        app.edit_col = int(clamp(app.edit_col, 0, len(circ.columns) - 1))
        app.edit_row = int(clamp(app.edit_row, 0, circ.n - 1))
        app.selected_q = int(clamp(app.selected_q, 0, circ.n - 1))

        # Render
        stdscr.erase()
        Hh, Ww = stdscr.getmaxyx()

        if Hh < 26 or Ww < 90:
            safe_addstr(stdscr, 0, 0, "Resize terminal to ~90x26 for the full builder UI.", curses.A_BOLD)
            safe_addstr(stdscr, 2, 0, f"Current: {Ww}x{Hh}", curses.A_DIM)
            stdscr.refresh()
            time.sleep(0.05)
            continue

        # Compute columns visible for scroll management
        label_w = 4
        cell_w = 6
        content_w = max(0, Ww - label_w - 1)
        vis_cols = max(1, content_w // cell_w)
        # keep scroll so focus target is visible
        focus_target = app.edit_col if app.focus == "edit" else app.play_col
        adjust_scroll_to_include(app, vis_cols, focus_target, len(circ.columns))

        # Header lines
        focus_txt = "EDIT" if app.focus == "edit" else "PLAY"
        hdr = (f"QC Circuit Builder | focus={focus_txt} | n={circ.n} cols={len(circ.columns)} "
               f"| play={app.play_col} | edit=({app.edit_row},{app.edit_col}) | plane={app.plane} metric={app.metric_mode} "
               f"| shots={'ON' if app.show_shots else 'OFF'}")
        safe_addstr(stdscr, 0, 0, hdr[:Ww-1], curses.A_BOLD)

        keys = "TAB focus | ? help | 1..4 templates | S save | L load | N new | u/U undo/redo | q quit"
        safe_addstr(stdscr, 1, 0, keys[:Ww-1], curses.A_DIM)

        if app.status_msg:
            safe_addstr(stdscr, 2, 0, app.status_msg[:Ww-1], c_met | curses.A_BOLD)
        else:
            safe_addstr(stdscr, 2, 0, " " * (Ww - 1))

        # Circuit area
        circuit_y = 3
        y_after_circ = draw_circuit(
            stdscr,
            y0=circuit_y,
            x0=0,
            w=Ww,
            circ=circ,
            play_col=app.play_col,
            edit_row=app.edit_row,
            edit_col=app.edit_col,
            scroll_col=app.scroll_col,
            focus=app.focus,
            pending=app.pending,
            c_hdr=c_hdr,
        )

        # Panels area
        panel_y = y_after_circ + 1

        # Separator vertical
        left_w = max(54, int(Ww * 0.58))
        right_x = left_w + 1
        right_w = Ww - right_x - 1
        for yy in range(panel_y, Hh - 1):
            safe_addstr(stdscr, yy, left_w, "│", curses.A_DIM)

        # Build state at play head
        psi = state_after_playhead(circ, app.play_col)
        p = probs_from_state(psi)

        # Shots
        if app.show_shots:
            dim = 1 << circ.n
            if shotwin.dim != dim:
                shotwin = ShotWindow.create(dim=dim, window=app.shots_window)
            outcomes = rng.choice(dim, size=app.shots_per_tick, p=p)
            shotwin.push(outcomes)

        # Left: amplitudes
        safe_addstr(stdscr, panel_y, 2, "Truth: complex amplitudes", c_hdr | curses.A_BOLD)
        draw_phase_legend(stdscr, panel_y + 1, 2, phase_attr)
        amp_y = panel_y + 3
        amp_h = max(10, Hh - amp_y - (6 if app.show_shots else 2))
        draw_amp_list(stdscr, amp_y, 2, left_w - 4, amp_h, psi, circ.n, phase_attr, topk=app.topk, sort_by_prob=app.sort_by_prob)

        if app.show_shots:
            f = shotwin.freqs()
            bot_y = amp_y + amp_h + 1
            safe_addstr(stdscr, bot_y, 2, f"Shots (window={shotwin.window}, filled={shotwin.filled})", curses.A_BOLD)
            bot_y += 1
            idxs = np.argsort(-p)[:min(8, len(p))]
            for i in idxs:
                if bot_y >= Hh - 1:
                    break
                safe_addstr(stdscr, bot_y, 2, f"{basis_label(int(i), circ.n):<{circ.n+3}}  true={p[i]:.3f}  meas={f[i]:.3f}", c_shot)
                bot_y += 1

        # Right: Bloch + entanglement
        safe_addstr(stdscr, panel_y, right_x + 1, "Local + Entanglement", c_hdr | curses.A_BOLD)
        rho = reduced_rho_1q(psi, circ.n, app.selected_q)
        bx, by, bz = bloch_from_rho(rho)
        pur = purity(rho)
        ent_meter = clamp(2.0 * (1.0 - pur), 0.0, 1.0)

        bloch_w = min(34, right_w - 2)
        bloch_h = 12
        draw_bloch_2d(stdscr, panel_y + 2, right_x + 1, bloch_w, bloch_h, app.plane, bx, by, bz)

        nx = right_x + 1 + bloch_w + 2
        ny = panel_y + 3
        if nx < right_x + right_w - 10:
            safe_addstr(stdscr, ny,     nx, f"q={app.selected_q}", curses.A_DIM)
            safe_addstr(stdscr, ny + 1, nx, f"⟨X⟩={bx:+.3f}", c_met)
            safe_addstr(stdscr, ny + 2, nx, f"⟨Y⟩={by:+.3f}", c_met)
            safe_addstr(stdscr, ny + 3, nx, f"⟨Z⟩={bz:+.3f}", c_met)
            safe_addstr(stdscr, ny + 5, nx, f"purity={pur:.3f}", curses.A_DIM)
            safe_addstr(stdscr, ny + 6, nx, "mixed≈", curses.A_DIM)
            draw_bar(stdscr, ny + 7, nx, max(8, min(18, right_x + right_w - nx - 12)), ent_meter, "", c_prob)

        ent_y = panel_y + 2 + bloch_h + 1
        if circ.n >= 2:
            if app.metric_mode == "corr":
                M = corr_matrix_Z(p, circ.n)
                draw_corr_heatmap(stdscr, ent_y, right_x + 1, right_w - 2, max(8, Hh - ent_y - 2), M, heat_attr,
                                  "Z-corr |corr(Zi,Zj)|")
                strength = M
            else:
                MI = np.zeros((circ.n, circ.n), dtype=np.float64)
                for i in range(circ.n):
                    for j in range(circ.n):
                        if i == j:
                            MI[i, j] = 1.0
                        else:
                            MI[i, j] = clamp(mutual_information_Z(p, circ.n, i, j), 0.0, 1.0)
                draw_corr_heatmap(stdscr, ent_y, right_x + 1, right_w - 2, max(8, Hh - ent_y - 2), MI, heat_attr,
                                  "Mutual info I(Zi;Zj) (clipped)")
                strength = MI

            edges = []
            for i in range(circ.n):
                for j in range(i + 1, circ.n):
                    edges.append((float(strength[i, j]), i, j))
            edges.sort(reverse=True)

            ly = min(Hh - 6, ent_y + 1 + min(circ.n + 2, 8))
            if ly < Hh - 1:
                safe_addstr(stdscr, ly, right_x + 1, "Strongest edges:", curses.A_BOLD)
                ly += 1
                for (wgt, i, j) in edges[:min(5, len(edges))]:
                    if ly >= Hh - 1:
                        break
                    link = "====" if wgt > 0.80 else "===" if wgt > 0.55 else "==" if wgt > 0.30 else "=" if wgt > 0.15 else "."
                    safe_addstr(stdscr, ly, right_x + 1, f"q{i} {link} q{j}  {wgt:0.3f}", heat_attr(wgt))
                    ly += 1

            if circ.n == 2:
                fids, best = bell_fidelities(psi)
                C = concurrence_pure_2q(psi)
                by2 = min(Hh - 8, ent_y + 2)
                safe_addstr(stdscr, by2, right_x + 1, "Bell meter (n=2)", curses.A_BOLD)
                name, fid = best[0], best[1]
                safe_addstr(stdscr, by2 + 1, right_x + 1, f"best≈{name}  fidelity={fid:.4f}", c_prob)
                draw_bar(stdscr, by2 + 2, right_x + 1, min(22, right_w - 14), fid, "F", c_prob)
                safe_addstr(stdscr, by2 + 3, right_x + 1, f"concurrence={C:.4f}", c_prob)
        else:
            safe_addstr(stdscr, ent_y, right_x + 1, "Entanglement: N/A for 1 qubit", curses.A_DIM)

        # Help overlay
        if app.show_help:
            box_w = min(Ww - 4, 88)
            box_h = min(Hh - 6, len(HELP_LINES) + 4)
            bx0 = 2
            by0 = 4
            # border
            for xx in range(bx0, bx0 + box_w):
                safe_addstr(stdscr, by0, xx, "─", curses.A_DIM)
                safe_addstr(stdscr, by0 + box_h - 1, xx, "─", curses.A_DIM)
            for yy in range(by0, by0 + box_h):
                safe_addstr(stdscr, yy, bx0, "│", curses.A_DIM)
                safe_addstr(stdscr, yy, bx0 + box_w - 1, "│", curses.A_DIM)
            safe_addstr(stdscr, by0, bx0, "┌", curses.A_DIM)
            safe_addstr(stdscr, by0, bx0 + box_w - 1, "┐", curses.A_DIM)
            safe_addstr(stdscr, by0 + box_h - 1, bx0, "└", curses.A_DIM)
            safe_addstr(stdscr, by0 + box_h - 1, bx0 + box_w - 1, "┘", curses.A_DIM)
            safe_addstr(stdscr, by0, bx0 + 2, " HELP (? to close) ", curses.A_BOLD)
            yy = by0 + 1
            for line in HELP_LINES[:box_h - 2]:
                safe_addstr(stdscr, yy, bx0 + 2, line[:box_w - 4], curses.A_DIM)
                yy += 1

        safe_addstr(stdscr, Hh - 1, 0, "EDIT: h/x/y/z/s/t r/e/w m c/v/p d I/D/A Q/K [ ] ENTER | PLAY: a +/- ←/→ r  | ? help"
                    [:Ww-1], curses.A_DIM)

        stdscr.refresh()
        time.sleep(0.016)


if __name__ == "__main__":
    curses.wrapper(main)
