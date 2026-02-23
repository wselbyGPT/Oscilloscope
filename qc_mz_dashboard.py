#!/usr/bin/env python3
import curses
import locale
import math
import time
from dataclasses import dataclass

import numpy as np


# ----------------------------
# Small helpers
# ----------------------------
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
    d = ((deg(a_rad) + 180) % 360) - 180
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:6.1f}°"

def cplx_phase(z: complex) -> float:
    return math.atan2(z.imag, z.real)

def basis_label(i: int, n: int) -> str:
    return "|" + format(i, f"0{n}b") + "⟩"


# ----------------------------
# Phase color palette (ANSI-ish via curses)
# ----------------------------
def init_phase_palette(stdscr):
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    colors = getattr(curses, "COLORS", 0)
    max_pairs = getattr(curses, "COLOR_PAIRS", 0)

    # 12 hues around a wheel (xterm-256 indices)
    xterm12 = [196, 208, 226, 118, 46, 48, 51, 39, 21, 93, 201, 198]
    # fallback basic colors (rough wheel)
    basic12 = [
        curses.COLOR_RED, curses.COLOR_YELLOW, curses.COLOR_YELLOW,
        curses.COLOR_GREEN, curses.COLOR_GREEN, curses.COLOR_CYAN,
        curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_BLUE,
        curses.COLOR_MAGENTA, curses.COLOR_MAGENTA, curses.COLOR_RED
    ]

    palette = []
    pair_base = 1
    want = 12
    can_make = min(want, max_pairs - pair_base) if max_pairs else want
    can_make = max(0, can_make)

    for k in range(can_make):
        fg = xterm12[k] if colors >= 256 else basic12[k]
        try:
            curses.init_pair(pair_base + k, fg, -1)
            palette.append(curses.color_pair(pair_base + k))
        except curses.error:
            palette.append(0)

    if not palette:
        palette = [0]

    def phase_attr(angle_rad: float) -> int:
        a = wrap_pi(angle_rad)
        t = (a + math.pi) / (2 * math.pi)  # 0..1
        idx = int(t * len(palette)) % len(palette)
        return palette[idx]

    return phase_attr, len(palette)


# ----------------------------
# Rolling shot window
# ----------------------------
@dataclass
class ShotWindow:
    window: int
    ring: np.ndarray
    pos: int
    filled: int
    counts: np.ndarray  # [count0, count1]

    @classmethod
    def create(cls, window: int):
        ring = np.full(window, -1, dtype=np.int16)
        counts = np.zeros(2, dtype=np.int32)
        return cls(window=window, ring=ring, pos=0, filled=0, counts=counts)

    def clear(self):
        self.ring.fill(-1)
        self.counts.fill(0)
        self.pos = 0
        self.filled = 0

    def push(self, outcomes: np.ndarray):
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

    def freqs(self):
        if self.filled <= 0:
            return 0.0, 0.0
        f0 = float(self.counts[0]) / float(self.filled)
        f1 = float(self.counts[1]) / float(self.filled)
        return f0, f1


# ----------------------------
# Mach–Zehnder (H → phase → H)
# Use 1-qubit statevector at each stage.
# ----------------------------
H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

def phase_gate(phi: float):
    # relative phase on |1> path (|0> untouched)
    return np.array([[1.0 + 0j, 0j], [0j, np.exp(1j * phi)]], dtype=np.complex128)

def stage_states(phi: float):
    """
    Returns:
      psi0: input |0>
      psi1: after first H
      psi2: after phase
      psi3: after second H
    """
    psi0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    psi1 = H @ psi0
    psi2 = phase_gate(phi) @ psi1
    psi3 = H @ psi2
    return psi0, psi1, psi2, psi3

def probs_from_state(psi: np.ndarray):
    p = np.abs(psi) ** 2
    s = float(np.sum(p))
    if s > 0:
        p = p / s
    return float(p[0]), float(p[1])


# ----------------------------
# ASCII wave rendering (two-path)
# ----------------------------
WAVE_CHARS = " .:-=+*#%@"

def wave_char(phase):
    s = math.sin(phase)
    t = (s + 1.0) * 0.5  # 0..1
    idx = int(t * (len(WAVE_CHARS) - 1))
    return WAVE_CHARS[idx]

def draw_wave_line(stdscr, y, x0, x1, base_phase, t, phase_attr, amp=1.0):
    if x1 <= x0:
        return
    k = 0.45  # spatial frequency
    w = 2.8   # temporal frequency
    for x in range(x0, x1):
        local = base_phase + (k * (x - x0)) - (w * t)
        if amp <= 1e-9:
            ch = " "
            attr = 0
        else:
            ch = wave_char(local)
            attr = phase_attr(wrap_pi(local))
        safe_addstr(stdscr, y, x, ch, attr)


# ----------------------------
# Amplitude/phase panel rendering
# ----------------------------
def draw_phase_legend(stdscr, y, x, width, phase_attr, bins=12):
    # small strip of colored blocks
    if width < bins + 10:
        return
    safe_addstr(stdscr, y, x, "phase:", curses.A_BOLD)
    x0 = x + 7
    for k in range(bins):
        ang = -math.pi + (2 * math.pi) * (k / bins)
        safe_addstr(stdscr, y, x0 + k, "█", phase_attr(ang))
    safe_addstr(stdscr, y, x0 + bins + 1, "-π", curses.A_DIM)
    safe_addstr(stdscr, y, x0 + bins + 5, "0", curses.A_DIM)
    safe_addstr(stdscr, y, x0 + bins + 8, "+π", curses.A_DIM)

def draw_amp_table(stdscr, y, x, width, title, psi, phase_attr):
    """
    Draws a tiny 2-row amplitude table for |0>, |1>.
    Bars are magnitude, colored by amplitude phase.
    """
    if width < 34:
        safe_addstr(stdscr, y, x, title[:max(0, width-1)], curses.A_BOLD)
        return y + 1

    safe_addstr(stdscr, y, x, title[:width-1], curses.A_BOLD)
    y += 1

    # header
    header = "basis    |amp|     phase      Re        Im     bar"
    safe_addstr(stdscr, y, x, header[:width-1], curses.A_UNDERLINE)
    y += 1

    mags = np.abs(psi)
    max_mag = float(np.max(mags)) if mags.size else 1.0
    if max_mag <= 0:
        max_mag = 1.0

    bar_w = max(8, width - len("basis    |amp|     phase      Re        Im     ") - 2)
    for i in range(2):
        amp = psi[i]
        mag = float(abs(amp))
        ph = cplx_phase(amp)
        re = float(amp.real)
        im = float(amp.imag)

        row = f"{basis_label(i,1):<7} {mag:7.4f}  {fmt_deg(ph):>9}  {re:8.4f}  {im:8.4f}  "
        safe_addstr(stdscr, y, x, row[:width-1])

        frac = clamp(mag / max_mag, 0.0, 1.0)
        fill = int(frac * bar_w)
        attr = phase_attr(ph)
        safe_addstr(stdscr, y, x + len(row), "█" * fill, attr)
        y += 1

    return y + 1


# ----------------------------
# Probability & histogram bars
# ----------------------------
def draw_bar(stdscr, y, x, w, frac, label, attr=0):
    frac = clamp(frac, 0.0, 1.0)
    fill = int(frac * w)
    safe_addstr(stdscr, y, x, f"{label} ", curses.A_BOLD)
    x2 = x + len(label) + 1
    safe_addstr(stdscr, y, x2, "█" * fill, attr)
    safe_addstr(stdscr, y, x2 + fill, " " * (w - fill))
    safe_addstr(stdscr, y, x2 + w + 1, f"{frac:6.3f}")


# ----------------------------
# Full dashboard render
# ----------------------------
def render_dashboard(
    stdscr,
    phi,
    auto_phi,
    speed,
    running,
    show_shots,
    shots_per_tick,
    shotwin,
    rng,
    phase_attr,
    t,
):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # Layout: left diagram + right amplitude truth
    left_w = max(44, int(w * 0.56))
    right_x = left_w + 1
    right_w = w - right_x - 1

    # Header
    hdr = "Two-Path Interference Dashboard | Left: paths | Right: amplitudes (truth)"
    safe_addstr(stdscr, 0, 0, hdr[:w-1], curses.A_BOLD)
    help1 = "keys: q quit | space pause | a auto-φ | ←/→ adjust φ | +/- speed | x shots | [ ] shots/tick | c clear shots | r reset φ"
    safe_addstr(stdscr, 1, 0, help1[:w-1], curses.A_DIM)

    # States
    _, psi1, psi2, psi3 = stage_states(phi)
    p0, p1 = probs_from_state(psi3)

    status = f"φ={phi:6.3f} rad ({fmt_deg(phi)}) | auto={auto_phi} | speed={speed:4.2f} | {'RUN' if running else 'PAUSE'} | shots={'ON' if show_shots else 'OFF'}"
    safe_addstr(stdscr, 2, 0, status[:w-1])

    # ---------------- Left: diagram ----------------
    top_y = 4
    mid_y = top_y + 3
    bot_y = top_y + 6

    # Diagram x positions inside left panel
    x_left = 2
    x_split = min(left_w - 30, 16)
    x_phase = min(left_w - 22, x_split + 16)
    x_merge = min(left_w - 16, x_phase + 14)
    x_out = min(left_w - 4, x_merge + 10)

    safe_addstr(stdscr, mid_y, x_left, "SRC", curses.A_BOLD)
    safe_addstr(stdscr, mid_y, x_split, "[H]", curses.A_BOLD)
    safe_addstr(stdscr, top_y, x_merge, "[H]", curses.A_BOLD)
    safe_addstr(stdscr, bot_y, x_merge, "[H]", curses.A_BOLD)

    # Source connector
    safe_addstr(stdscr, mid_y, x_left + 4, "-" * max(0, (x_split - (x_left + 4))), curses.A_DIM)

    # Split/merge glyphs
    safe_addstr(stdscr, mid_y - 1, x_split + 2, "/")
    safe_addstr(stdscr, mid_y + 1, x_split + 2, "\\")
    safe_addstr(stdscr, top_y, x_split + 3, ">", curses.A_DIM)
    safe_addstr(stdscr, bot_y, x_split + 3, ">", curses.A_DIM)

    safe_addstr(stdscr, top_y, x_merge - 1, "<", curses.A_DIM)
    safe_addstr(stdscr, bot_y, x_merge - 1, "<", curses.A_DIM)
    safe_addstr(stdscr, mid_y - 1, x_merge + 2, "\\")
    safe_addstr(stdscr, mid_y + 1, x_merge + 2, "/")

    # Paths
    safe_addstr(stdscr, top_y - 1, x_left + 4, "upper path (|0⟩ branch)", curses.A_DIM)
    draw_wave_line(stdscr, top_y, x_split + 4, x_merge - 1, 0.0, t, phase_attr)

    safe_addstr(stdscr, bot_y - 1, x_left + 4, f"lower path (|1⟩ branch)  phase=φ {fmt_deg(phi)}", curses.A_DIM)
    draw_wave_line(stdscr, bot_y, x_split + 4, x_phase - 2, phi, t, phase_attr)
    safe_addstr(stdscr, bot_y, x_phase - 1, "[φ]", curses.A_BOLD)
    draw_wave_line(stdscr, bot_y, x_phase + 3, x_merge - 1, phi, t, phase_attr)

    # Outputs to detectors
    safe_addstr(stdscr, top_y, x_merge + 4, "-" * max(0, x_out - (x_merge + 4)), curses.A_DIM)
    safe_addstr(stdscr, bot_y, x_merge + 4, "-" * max(0, x_out - (x_merge + 4)), curses.A_DIM)
    safe_addstr(stdscr, top_y, x_out, "D0", curses.A_BOLD)
    safe_addstr(stdscr, bot_y, x_out, "D1", curses.A_BOLD)

    # Prob bars (left)
    bar_y = bot_y + 2
    if bar_y < h - 2:
        bw = max(10, min(36, left_w - 16))
        draw_bar(stdscr, bar_y, x_left, bw, p0, "P(D0)")
        draw_bar(stdscr, bar_y + 1, x_left, bw, p1, "P(D1)")
        safe_addstr(stdscr, bar_y + 2, x_left, "Theory: P(D0)=cos^2(φ/2),  P(D1)=sin^2(φ/2)", curses.A_DIM)

    # Optional rolling shots (left)
    if show_shots:
        outcomes = rng.choice(2, size=shots_per_tick, p=np.array([p0, p1], dtype=np.float64))
        shotwin.push(outcomes)
        f0, f1 = shotwin.freqs()

        sy = bar_y + 4
        if sy < h - 2:
            safe_addstr(stdscr, sy, x_left, f"Shots (window={shotwin.window}, filled={shotwin.filled})", curses.A_BOLD)
            bw2 = max(10, min(36, left_w - 16))
            draw_bar(stdscr, sy + 1, x_left, bw2, f0, "freq(D0)")
            draw_bar(stdscr, sy + 2, x_left, bw2, f1, "freq(D1)")
            tv = 0.5 * (abs(f0 - p0) + abs(f1 - p1))
            safe_addstr(stdscr, sy + 3, x_left, f"TV distance ≈ {tv:.4f} (sampling error)", curses.A_DIM)

    # Separator
    for yy in range(3, h - 1):
        safe_addstr(stdscr, yy, left_w, "│", curses.A_DIM)

    # ---------------- Right: amplitude/phase truth ----------------
    if right_w >= 20:
        ry = 3
        safe_addstr(stdscr, ry, right_x, "Amplitude Truth (|amp| + phase)".ljust(right_w), curses.A_BOLD)
        ry += 1
        draw_phase_legend(stdscr, ry, right_x, right_w, phase_attr)
        ry += 2

        # Show stage tables
        ry = draw_amp_table(stdscr, ry, right_x, right_w, "After BS1 (H):  ψ1 = H|0⟩", psi1, phase_attr)
        ry = draw_amp_table(stdscr, ry, right_x, right_w, "After Phase:    ψ2 = phase(φ)·ψ1", psi2, phase_attr)
        ry = draw_amp_table(stdscr, ry, right_x, right_w, "After BS2 (H):  ψ3 = H·ψ2  (detectors)", psi3, phase_attr)

        # A tiny “interference explanation” line
        if ry < h - 2:
            safe_addstr(
                stdscr,
                ry,
                right_x,
                "Interference happens at BS2: amplitudes add/subtract (a±b)/√2, so phase controls cancellation.",
                curses.A_DIM,
            )

    # Footer
    safe_addstr(stdscr, h - 1, 0, "Try φ=0° (all D0), φ=180° (all D1). Watch right panel phases flip the left outcomes.", curses.A_DIM)
    stdscr.refresh()


# ----------------------------
# Main loop
# ----------------------------
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    phase_attr, _ = init_phase_palette(stdscr)

    phi = 0.0
    auto_phi = True
    speed = 1.0
    running = True

    show_shots = False
    shots_per_tick = 200
    shotwin = ShotWindow.create(window=4000)
    rng = np.random.default_rng(1)

    last = time.time()
    t0 = last

    while True:
        now = time.time()
        dt = now - last
        last = now
        t = now - t0

        ch = -1
        try:
            ch = stdscr.getch()
        except curses.error:
            pass

        if ch != -1:
            if ch in (ord('q'), ord('Q')):
                break
            elif ch == ord(' '):
                running = not running
            elif ch in (ord('a'), ord('A')):
                auto_phi = not auto_phi
            elif ch in (ord('+'), ord('=')):
                speed = min(10.0, speed * 1.25)
            elif ch in (ord('-'), ord('_')):
                speed = max(0.05, speed / 1.25)
            elif ch in (curses.KEY_LEFT, ord('h')):
                phi -= 0.10
                auto_phi = False
            elif ch in (curses.KEY_RIGHT, ord('l')):
                phi += 0.10
                auto_phi = False
            elif ch in (ord('x'), ord('X')):
                show_shots = not show_shots
                shotwin.clear()
            elif ch == ord('['):
                shots_per_tick = max(1, int(shots_per_tick / 1.5))
            elif ch == ord(']'):
                shots_per_tick = min(20000, int(shots_per_tick * 1.5) + 1)
            elif ch in (ord('c'), ord('C')):
                shotwin.clear()
            elif ch in (ord('r'), ord('R')):
                phi = 0.0
                auto_phi = True
                shotwin.clear()

        if running and auto_phi:
            phi += dt * speed

        phi = wrap_pi(phi)

        render_dashboard(
            stdscr=stdscr,
            phi=phi,
            auto_phi=auto_phi,
            speed=speed,
            running=running,
            show_shots=show_shots,
            shots_per_tick=shots_per_tick,
            shotwin=shotwin,
            rng=rng,
            phase_attr=phase_attr,
            t=t,
        )

        time.sleep(0.016)  # ~60 fps


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(main)
