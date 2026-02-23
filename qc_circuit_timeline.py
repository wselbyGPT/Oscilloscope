#!/usr/bin/env python3
import argparse
import curses
import locale
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# Helpers
# ============================================================
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


# ============================================================
# Color palettes
# ============================================================
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

    # Some basics
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

    return phase_attr, heat_attr, c_hdr, c_met, c_prob, c_shot


# ============================================================
# Quantum simulation (statevector)  q=0 is LSB
# ============================================================
def gate_H():
    return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

def gate_X():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

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

def probs_from_state(psi: np.ndarray) -> np.ndarray:
    p = np.abs(psi) ** 2
    s = float(np.sum(p))
    if s > 0:
        p = p / s
    return p


# ============================================================
# Reduced density matrix for 1 qubit + Bloch
# ============================================================
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


# ============================================================
# Entanglement-ish metrics
# ============================================================
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


# ============================================================
# Circuit representation
# ============================================================
@dataclass
class Op:
    kind: str                    # "H","X","RZ","CNOT"
    q: Optional[int] = None
    control: Optional[int] = None
    target: Optional[int] = None
    param: Optional[str] = None  # e.g. "theta" for RZ(theta)

def make_demo_circuit(n: int) -> List[List[Op]]:
    """
    A GHZ-like build + phase + uncompute:
      col0: H q0
      cols: CNOT chain 0->1->2->...
      col:  RZ(theta) on q0
      cols: CNOT chain reversed (disentangle)
      col:  H q0
    """
    cols: List[List[Op]] = []
    cols.append([Op("H", q=0)])
    for i in range(n - 1):
        cols.append([Op("CNOT", control=i, target=i + 1)])
    cols.append([Op("RZ", q=0, param="theta")])
    for i in reversed(range(n - 1)):
        cols.append([Op("CNOT", control=i, target=i + 1)])
    cols.append([Op("H", q=0)])
    return cols

def apply_column(psi: np.ndarray, n: int, col_ops: List[Op], theta: float) -> np.ndarray:
    out = psi
    for op in col_ops:
        if op.kind == "H":
            out = apply_1q(out, n, op.q, gate_H())
        elif op.kind == "X":
            out = apply_1q(out, n, op.q, gate_X())
        elif op.kind == "RZ":
            th = theta if op.param == "theta" else 0.0
            out = apply_1q(out, n, op.q, gate_RZ(th))
        elif op.kind == "CNOT":
            out = apply_cnot(out, n, op.control, op.target)
    return out

def state_after_step(n: int, circuit: List[List[Op]], step_col: int, theta: float) -> np.ndarray:
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0j
    for c in range(step_col + 1):
        psi = apply_column(psi, n, circuit[c], theta)
    return psi


# ============================================================
# Drawing primitives
# ============================================================
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

    # circle
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


# ============================================================
# Circuit timeline renderer (Composer-ish)
# ============================================================
def gate_cell(label: str) -> str:
    """
    Fixed-width cell (5 chars). Keep wire-ish look.
    """
    if label == "H":
        return "─[H]─"
    if label == "X":
        return "─[X]─"
    if label in ("Rz", "Ry", "Rx"):
        # slightly different but fits
        s = f"[{label}]"
        return (s + "─") if len(s) == 4 else s[:5].ljust(5)
    return "─────"

def draw_circuit(
    stdscr,
    y0: int,
    x0: int,
    w: int,
    n: int,
    circuit: List[List[Op]],
    current_col: int,
    scroll_col: int,
    theta: float,
    c_hdr=0,
):
    """
    Draw a circuit timeline like IBM composer:
      - row labels q0..q{n-1}
      - columns with gates
      - highlight current_col
      - horizontal scrolling by scroll_col
    """
    label_w = 4  # "q0: "
    cell_w = 6   # 5 chars content + 1 space
    content_w = max(0, w - label_w - 1)
    vis_cols = max(1, content_w // cell_w)

    total_cols = len(circuit)
    start = clamp(scroll_col, 0, max(0, total_cols - vis_cols))
    end = min(total_cols, start + vis_cols)

    # top header row: column indices
    safe_addstr(stdscr, y0, x0, " " * label_w, 0)
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        txt = f"{ci:>2}"
        attr = curses.A_REVERSE if ci == current_col else curses.A_DIM
        safe_addstr(stdscr, y0, x, txt, attr)

    # optional param row: show theta under RZ columns
    y_param = y0 + 1
    safe_addstr(stdscr, y_param, x0, " " * label_w, 0)
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        has_rz = any(op.kind == "RZ" for op in circuit[ci])
        if has_rz:
            s = f"θ={theta:+.2f}"
            s = s[:5].ljust(5)
            attr = curses.A_REVERSE if ci == current_col else curses.A_DIM
            safe_addstr(stdscr, y_param, x, s, attr)

    # precompute per column: CNOT endpoints for vertical wires
    cnots: List[Tuple[int, int, int]] = []  # (col, c, t)
    for ci, ops in enumerate(circuit):
        for op in ops:
            if op.kind == "CNOT":
                cnots.append((ci, op.control, op.target))

    # draw rows
    y_rows = y0 + 2
    for q in range(n):
        y = y_rows + q
        safe_addstr(stdscr, y, x0, f"q{q}:".ljust(label_w), curses.A_BOLD | c_hdr)

        for ci in range(start, end):
            x = x0 + label_w + (ci - start) * cell_w
            attr = curses.A_REVERSE if ci == current_col else 0
            # default wire
            safe_addstr(stdscr, y, x, "─────", attr)

    # overlay 1q gates
    for ci in range(start, end):
        x = x0 + label_w + (ci - start) * cell_w
        col_attr = curses.A_REVERSE if ci == current_col else 0
        for op in circuit[ci]:
            if op.kind in ("H", "X", "RZ"):
                y = y_rows + op.q
                label = "H" if op.kind == "H" else "X" if op.kind == "X" else "Rz"
                safe_addstr(stdscr, y, x, gate_cell(label), col_attr)

    # overlay CNOTs (● control, X target, │ between)
    for (ci, c, t) in cnots:
        if ci < start or ci >= end:
            continue
        x = x0 + label_w + (ci - start) * cell_w
        col_attr = curses.A_REVERSE if ci == current_col else 0
        y_c = y_rows + c
        y_t = y_rows + t
        y_min, y_max = (y_c, y_t) if y_c <= y_t else (y_t, y_c)

        # vertical line
        for yy in range(y_min + 1, y_max):
            safe_addstr(stdscr, yy, x + 2, "│", col_attr)

        # control + target
        safe_addstr(stdscr, y_c, x + 2, "●", col_attr | curses.A_BOLD)
        safe_addstr(stdscr, y_t, x + 2, "X", col_attr | curses.A_BOLD)

        # reinforce wire around marker
        safe_addstr(stdscr, y_c, x, "──", col_attr)
        safe_addstr(stdscr, y_c, x + 3, "──", col_attr)
        safe_addstr(stdscr, y_t, x, "──", col_attr)
        safe_addstr(stdscr, y_t, x + 3, "──", col_attr)

    # footer line in circuit area
    info = f"circuit cols {total_cols} | view {start}..{end-1} | scroll < >"
    safe_addstr(stdscr, y_rows + n + 1, x0, info[:w-1], curses.A_DIM)


# ============================================================
# Panels: amplitudes + entanglement heatmap
# ============================================================
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

        label = basis_label(int(i), n)
        label_clip = label[: max(0, (n + 3))]
        row = f"{label_clip:<{n+3}} {mag:6.4f} {prob:7.4f} {fmt_deg(ph):>9}  "
        safe_addstr(stdscr, y, x0, row[:w-1])

        frac = clamp(mag / max_mag, 0.0, 1.0)
        fill = int(frac * bar_w)
        attr = phase_attr(ph)
        safe_addstr(stdscr, y, bar_x, "█" * fill, attr)
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
            ch = shade_char(t)
            safe_addstr(stdscr, y, x, ch, heat_attr(t))
            safe_addstr(stdscr, y, x + 1, " ")
            x += 2
        y += 1


# ============================================================
# Main TUI
# ============================================================
@dataclass
class Config:
    n: int
    auto_play: bool = True
    running: bool = True
    col_rate: float = 2.0  # columns/sec
    theta: float = 0.0
    auto_theta: bool = True
    theta_speed: float = 1.2
    current_col: int = 0
    scroll_col: int = 0
    selected_q: int = 0
    plane: str = "XY"
    metric_mode: str = "corr"  # corr or mi
    topk: int = 16
    sort_by_prob: bool = True


def render(stdscr, cfg: Config, circuit: List[List[Op]], phase_attr, heat_attr, c_hdr, c_met, c_prob):
    stdscr.erase()
    Hh, Ww = stdscr.getmaxyx()

    total_cols = len(circuit)
    cfg.current_col = int(clamp(cfg.current_col, 0, total_cols - 1))

    # Build state for current step
    psi = state_after_step(cfg.n, circuit, cfg.current_col, cfg.theta)
    p = probs_from_state(psi)

    # Metrics: local Bloch for selected qubit
    rho = reduced_rho_1q(psi, cfg.n, cfg.selected_q)
    bx, by, bz = bloch_from_rho(rho)
    pur = purity(rho)
    ent_meter = clamp(2.0 * (1.0 - pur), 0.0, 1.0)  # quick “how mixed” proxy

    # Header
    status = "RUN" if cfg.running else "PAUSE"
    ap = "AUTO" if cfg.auto_play else "MAN"
    thm = "AUTO" if cfg.auto_theta else "MAN"
    hdr = (f"Circuit Timeline Stepper | n={cfg.n} | col {cfg.current_col+1}/{total_cols} | {status}/{ap} "
           f"| θ={cfg.theta:+.3f} ({fmt_deg(cfg.theta)}) {thm} | plane={cfg.plane} | q={cfg.selected_q} | metric={cfg.metric_mode}")
    safe_addstr(stdscr, 0, 0, hdr[:Ww-1], curses.A_BOLD)

    help1 = ("keys: q quit | space pause | a autoplay | ←/→ step | +/- speed | </> scroll | "
             "t topK | s sort | k qubit | p plane | m metric | ,/. θ | z auto-θ | r reset")
    safe_addstr(stdscr, 1, 0, help1[:Ww-1], curses.A_DIM)

    # Layout
    circuit_h = cfg.n + 4  # col idx + theta row + n rows + footer
    circuit_y = 3

    # Top: circuit
    draw_circuit(
        stdscr,
        y0=circuit_y,
        x0=0,
        w=Ww,
        n=cfg.n,
        circuit=circuit,
        current_col=cfg.current_col,
        scroll_col=cfg.scroll_col,
        theta=cfg.theta,
        c_hdr=c_hdr,
    )

    # Under circuit: panels (left amplitudes, right bloch + entanglement)
    panel_y = circuit_y + circuit_h + 1
    if panel_y >= Hh - 2:
        safe_addstr(stdscr, Hh - 1, 0, "Resize terminal taller for panels.", curses.A_DIM)
        stdscr.refresh()
        return

    left_w = max(50, int(Ww * 0.58))
    right_x = left_w + 1
    right_w = Ww - right_x - 1

    # separator
    for yy in range(panel_y, Hh - 1):
        safe_addstr(stdscr, yy, left_w, "│", curses.A_DIM)

    # left: amplitudes
    safe_addstr(stdscr, panel_y, 2, "Truth: complex amplitudes", c_hdr | curses.A_BOLD)
    draw_phase_legend(stdscr, panel_y + 1, 2, phase_attr)
    amp_y = panel_y + 3
    amp_h = max(8, Hh - amp_y - 2)
    draw_amp_list(stdscr, amp_y, 2, left_w - 4, amp_h, psi, cfg.n, phase_attr, topk=cfg.topk, sort_by_prob=cfg.sort_by_prob)

    # right: bloch
    safe_addstr(stdscr, panel_y, right_x + 1, "Local: Bloch + entanglement", c_hdr | curses.A_BOLD)
    bloch_w = min(34, right_w - 2)
    bloch_h = 12
    draw_bloch_2d(stdscr, panel_y + 2, right_x + 1, bloch_w, bloch_h, cfg.plane, bx, by, bz)

    nx = right_x + 1 + bloch_w + 2
    ny = panel_y + 3
    if nx < right_x + right_w - 8:
        safe_addstr(stdscr, ny,     nx, f"⟨X⟩={bx:+.3f}", c_met)
        safe_addstr(stdscr, ny + 1, nx, f"⟨Y⟩={by:+.3f}", c_met)
        safe_addstr(stdscr, ny + 2, nx, f"⟨Z⟩={bz:+.3f}", c_met)
        safe_addstr(stdscr, ny + 4, nx, f"purity={pur:.3f}", curses.A_DIM)
        safe_addstr(stdscr, ny + 5, nx, "mixed≈", curses.A_DIM)
        draw_bar(stdscr, ny + 6, nx, max(8, min(18, right_x + right_w - nx - 12)), ent_meter, "", c_prob)

    # entanglement view
    ent_y = panel_y + 2 + bloch_h + 1
    if ent_y < Hh - 2:
        if cfg.metric_mode == "corr":
            M = corr_matrix_Z(p, cfg.n)
            draw_corr_heatmap(stdscr, ent_y, right_x + 1, right_w - 2, max(6, Hh - ent_y - 2), M, heat_attr,
                              "Z-correlation |corr(Zi,Zj)|")
            strength = M
        else:
            MI = np.zeros((cfg.n, cfg.n), dtype=np.float64)
            for i in range(cfg.n):
                for j in range(cfg.n):
                    if i == j:
                        MI[i, j] = 1.0
                    else:
                        MI[i, j] = clamp(mutual_information_Z(p, cfg.n, i, j), 0.0, 1.0)
            draw_corr_heatmap(stdscr, ent_y, right_x + 1, right_w - 2, max(6, Hh - ent_y - 2), MI, heat_attr,
                              "Mutual info I(Zi;Zj) (bits, clipped)")
            strength = MI

        # strongest edges list (graph-ish)
        edges = []
        for i in range(cfg.n):
            for j in range(i + 1, cfg.n):
                edges.append((float(strength[i, j]), i, j))
        edges.sort(reverse=True)

        ly = min(Hh - 6, ent_y + 1 + min(cfg.n + 2, 8))
        if ly < Hh - 1:
            safe_addstr(stdscr, ly, right_x + 1, "Strongest edges:", curses.A_BOLD)
            ly += 1
            for (wgt, i, j) in edges[:min(5, len(edges))]:
                if ly >= Hh - 1:
                    break
                link = "====" if wgt > 0.80 else "===" if wgt > 0.55 else "==" if wgt > 0.30 else "=" if wgt > 0.15 else "."
                safe_addstr(stdscr, ly, right_x + 1, f"q{i} {link} q{j}  {wgt:0.3f}", heat_attr(wgt))
                ly += 1

        # Bell meter for n=2
        if cfg.n == 2:
            fids, best = bell_fidelities(psi)
            C = concurrence_pure_2q(psi)
            by2 = min(Hh - 9, ent_y + 2)
            safe_addstr(stdscr, by2, right_x + 1, "Bell meter (n=2)", curses.A_BOLD)
            name, fid = best[0], best[1]
            safe_addstr(stdscr, by2 + 1, right_x + 1, f"best≈{name}  fidelity={fid:.4f}", c_prob)
            draw_bar(stdscr, by2 + 2, right_x + 1, min(22, right_w - 14), fid, "F", c_prob)
            safe_addstr(stdscr, by2 + 3, right_x + 1, f"concurrence={C:.4f}", c_prob)

    stdscr.refresh()


def main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    phase_attr, heat_attr, c_hdr, c_met, c_prob, _c_shot = init_palettes(stdscr)

    cfg = Config(n=args.qubits)
    cfg.col_rate = args.speed
    cfg.theta_speed = args.theta_speed
    cfg.topk = args.topk

    circuit = make_demo_circuit(cfg.n)
    total_cols = len(circuit)

    last = time.time()
    acc = 0.0
    t0 = last

    while True:
        now = time.time()
        dt = now - last
        last = now
        t = now - t0

        # Input
        ch = -1
        try:
            ch = stdscr.getch()
        except curses.error:
            pass

        if ch != -1:
            if ch in (ord('q'), ord('Q')):
                break
            elif ch == ord(' '):
                cfg.running = not cfg.running
            elif ch in (ord('a'), ord('A')):
                cfg.auto_play = not cfg.auto_play
            elif ch in (ord('+'), ord('=')):
                cfg.col_rate = min(20.0, cfg.col_rate * 1.25)
            elif ch in (ord('-'), ord('_')):
                cfg.col_rate = max(0.2, cfg.col_rate / 1.25)
            elif ch in (curses.KEY_LEFT, ord('h')):
                cfg.auto_play = False
                cfg.current_col = max(0, cfg.current_col - 1)
            elif ch in (curses.KEY_RIGHT, ord('l')):
                cfg.auto_play = False
                cfg.current_col = min(total_cols - 1, cfg.current_col + 1)
            elif ch in (ord('<'), ord(',')):  # scroll left (note: ',' also used for theta if auto off; keep both)
                cfg.scroll_col = max(0, cfg.scroll_col - 1)
            elif ch in (ord('>'), ord('.')):  # scroll right
                cfg.scroll_col = min(max(0, total_cols - 1), cfg.scroll_col + 1)

            elif ch in (ord('k'), ord('K')):
                cfg.selected_q = (cfg.selected_q + 1) % cfg.n
            elif ch in (ord('p'), ord('P')):
                cfg.plane = "XZ" if cfg.plane == "XY" else "XY"
            elif ch in (ord('m'), ord('M')):
                cfg.metric_mode = "mi" if cfg.metric_mode == "corr" else "corr"

            elif ch in (ord('t'), ord('T')):
                # cycle: 8 -> 16 -> 32 -> all -> 8
                if cfg.topk == 8:
                    cfg.topk = 16
                elif cfg.topk == 16:
                    cfg.topk = 32
                elif cfg.topk == 32:
                    cfg.topk = 0
                else:
                    cfg.topk = 8
            elif ch in (ord('s'), ord('S')):
                cfg.sort_by_prob = not cfg.sort_by_prob

            elif ch in (ord('z'), ord('Z')):
                cfg.auto_theta = not cfg.auto_theta

            # theta manual adjust (when auto_theta off)
            elif ch in (ord('['),):
                cfg.auto_theta = False
                cfg.theta -= 0.10
            elif ch in (ord(']'),):
                cfg.auto_theta = False
                cfg.theta += 0.10

            elif ch in (ord('r'), ord('R')):
                cfg.current_col = 0
                cfg.scroll_col = 0
                cfg.theta = 0.0
                cfg.auto_play = True
                cfg.auto_theta = True
                cfg.running = True

        # Update theta
        if cfg.running:
            if cfg.auto_theta:
                cfg.theta = wrap_pi(math.sin(t * cfg.theta_speed) * math.pi)  # smooth sweep [-pi, pi]
            else:
                cfg.theta = wrap_pi(cfg.theta)

        # Autoplay column stepping
        if cfg.running and cfg.auto_play:
            acc += dt * cfg.col_rate
            while acc >= 1.0:
                acc -= 1.0
                cfg.current_col += 1
                if cfg.current_col >= total_cols:
                    cfg.current_col = 0

        render(stdscr, cfg, circuit, phase_attr, heat_attr, c_hdr, c_met, c_prob)
        time.sleep(0.016)


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    ap = argparse.ArgumentParser(description="ASCII/ANSI quantum circuit timeline stepper (curses)")
    ap.add_argument("--qubits", type=int, default=3, help="number of qubits (2..6 recommended)")
    ap.add_argument("--speed", type=float, default=2.0, help="autoplay speed (columns/sec)")
    ap.add_argument("--theta-speed", dest="theta_speed", type=float, default=1.2, help="theta sweep speed")
    ap.add_argument("--topk", type=int, default=16, help="top-K basis states shown (0 shows all)")
    args = ap.parse_args()

    if args.qubits < 1 or args.qubits > 8:
        raise SystemExit("Choose --qubits between 1 and 8 (best visuals 2..6).")

    curses.wrapper(main, args)
