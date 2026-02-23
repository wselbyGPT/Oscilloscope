#!/usr/bin/env python3
# fft_curses_viz.py
#
# Live ASCII/ANSI FFT visualizer (waveform + spectrum + spectrogram) using curses.
# Works well in WSL Ubuntu inside Windows Terminal (256-color).
#
# Keys:
#   q            quit
#   space        pause/resume
#   h            toggle help
#   1/2/3        select tone
#   ←/→          selected tone: frequency down/up
#   ↓/↑          selected tone: amplitude down/up
#   n            toggle noise
#   +/-          noise amplitude down/up
#   l            toggle log spectrum (dB) vs linear
#   w            toggle Hann window
#   r            reset to defaults

import curses
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ----------------------------
# Color palette (256-color HSV gradient)
# ----------------------------
def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def rgb_to_xterm256(r: float, g: float, b: float) -> int:
    # Quantize to 6x6x6 color cube (16..231)
    rq = int(round(r * 5))
    gq = int(round(g * 5))
    bq = int(round(b * 5))
    rq = min(5, max(0, rq))
    gq = min(5, max(0, gq))
    bq = min(5, max(0, bq))
    return 16 + 36 * rq + 6 * gq + bq


def build_palette(n_levels: int, use_256: bool) -> List[int]:
    if not use_256:
        # Fallback to basic colors (repeat if needed)
        basic = [curses.COLOR_BLUE, curses.COLOR_CYAN, curses.COLOR_GREEN,
                 curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_MAGENTA, curses.COLOR_WHITE]
        out = []
        for i in range(n_levels):
            out.append(basic[i % len(basic)])
        return out

    # HSV rainbow gradient
    cols = []
    for i in range(n_levels):
        h = i / max(1, (n_levels - 1))
        # Slightly avoid wrapping back to red at the very end
        h *= 0.90
        r, g, b = hsv_to_rgb(h, 1.0, 1.0)
        cols.append(rgb_to_xterm256(r, g, b))
    return cols


def init_color_pairs(n_levels: int) -> Tuple[List[int], int]:
    curses.start_color()
    try:
        curses.use_default_colors()
        default_bg = -1
    except curses.error:
        default_bg = curses.COLOR_BLACK

    use_256 = (getattr(curses, "COLORS", 0) >= 256)
    palette = build_palette(n_levels, use_256)

    max_pairs = getattr(curses, "COLOR_PAIRS", 64)
    # Reserve pair 1..n
    usable = min(n_levels, max_pairs - 1)
    pair_ids = []
    for i in range(usable):
        pid = i + 1
        fg = palette[i]
        try:
            curses.init_pair(pid, fg, default_bg)
        except curses.error:
            # Some terms reject 256 indices; fallback to white
            curses.init_pair(pid, curses.COLOR_WHITE, default_bg)
        pair_ids.append(pid)

    # Return list of pair IDs and actual levels
    return pair_ids, usable


# ----------------------------
# Signal model
# ----------------------------
@dataclass
class Tone:
    freq: float
    amp: float
    phase: float = 0.0


def gen_samples(t0: float, n: int, sr: float, tones: List[Tone], noise_amp: float) -> Tuple[np.ndarray, float]:
    t = t0 + np.arange(n) / sr
    x = np.zeros(n, dtype=np.float32)
    for tone in tones:
        x += tone.amp * np.sin(2.0 * math.pi * tone.freq * t + tone.phase).astype(np.float32)
    if noise_amp > 0:
        x += (noise_amp * np.random.normal(0.0, 1.0, size=n)).astype(np.float32)
    return x, t0 + n / sr


def hann_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


# ----------------------------
# Drawing helpers
# ----------------------------
CHARS = " .:-=+*#%@"
BLOCK = "█"


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def safe_addch(stdscr, y: int, x: int, ch: str, attr: int = 0):
    try:
        stdscr.addch(y, x, ch, attr)
    except curses.error:
        pass


def safe_addstr(stdscr, y: int, x: int, s: str, attr: int = 0):
    try:
        stdscr.addstr(y, x, s, attr)
    except curses.error:
        pass


def val_to_style(val01: float, pair_ids: List[int]) -> Tuple[str, int]:
    val01 = clamp(val01, 0.0, 1.0)
    n = len(pair_ids)
    if n <= 0:
        idx = int(val01 * (len(CHARS) - 1))
        return CHARS[idx], 0
    level = int(val01 * (n - 1))
    level = min(n - 1, max(0, level))
    idx = int(val01 * (len(CHARS) - 1))
    ch = CHARS[min(len(CHARS) - 1, max(0, idx))]
    return ch, curses.color_pair(pair_ids[level])


def draw_box(stdscr, y0: int, x0: int, h: int, w: int, title: str = ""):
    if h < 2 or w < 2:
        return
    # Simple border
    for x in range(x0, x0 + w):
        safe_addch(stdscr, y0, x, "─")
        safe_addch(stdscr, y0 + h - 1, x, "─")
    for y in range(y0, y0 + h):
        safe_addch(stdscr, y, x0, "│")
        safe_addch(stdscr, y, x0 + w - 1, "│")
    safe_addch(stdscr, y0, x0, "┌")
    safe_addch(stdscr, y0, x0 + w - 1, "┐")
    safe_addch(stdscr, y0 + h - 1, x0, "└")
    safe_addch(stdscr, y0 + h - 1, x0 + w - 1, "┘")

    if title and w > len(title) + 4:
        safe_addstr(stdscr, y0, x0 + 2, f"{title}")


def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def draw_waveform(stdscr, y0: int, x0: int, h: int, w: int, sig: np.ndarray, pair_ids: List[int]):
    if h < 3 or w < 5:
        return
    draw_box(stdscr, y0, x0, h, w, "Time domain")

    inner_h = h - 2
    inner_w = w - 2
    top = y0 + 1
    left = x0 + 1

    # Midline
    midy = top + inner_h // 2
    for x in range(left, left + inner_w):
        safe_addch(stdscr, midy, x, "·")

    # Resample signal to width
    n = len(sig)
    idx = np.linspace(0, n - 1, inner_w).astype(int)
    s = sig[idx]
    # Normalize for display
    peak = float(np.max(np.abs(s)) + 1e-9)
    s = (s / peak) * 0.95

    prev_x = left
    prev_y = midy - int(s[0] * (inner_h // 2 - 1))
    prev_y = min(top + inner_h - 1, max(top, prev_y))

    for i in range(1, inner_w):
        x = left + i
        y = midy - int(s[i] * (inner_h // 2 - 1))
        y = min(top + inner_h - 1, max(top, y))

        # Color based on magnitude
        val01 = float(abs(s[i]))
        ch, attr = val_to_style(val01, pair_ids)

        for lx, ly in bresenham_line(prev_x, prev_y, x, y):
            safe_addch(stdscr, ly, lx, ch, attr)
        prev_x, prev_y = x, y


def downsample_to_width(arr: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(arr) == width:
        return arr.astype(np.float32, copy=False)
    idx = np.linspace(0, len(arr) - 1, width).astype(int)
    return arr[idx].astype(np.float32, copy=False)


def mag_to_norm(mag: np.ndarray, log_scale: bool) -> np.ndarray:
    if len(mag) == 0:
        return mag
    if log_scale:
        mag_db = 20.0 * np.log10(mag + 1e-12)
        mag_db -= np.max(mag_db)
        mag_db = np.clip(mag_db, -80.0, 0.0)
        norm = (mag_db + 80.0) / 80.0
        return norm.astype(np.float32)
    else:
        m = float(np.max(mag) + 1e-12)
        return (mag / m).astype(np.float32)


def draw_spectrum(stdscr, y0: int, x0: int, h: int, w: int,
                  norm_bins: np.ndarray, pair_ids: List[int],
                  tone_freqs: List[float], sr: float):
    if h < 3 or w < 5:
        return
    draw_box(stdscr, y0, x0, h, w, "Frequency spectrum")

    inner_h = h - 2
    inner_w = w - 2
    top = y0 + 1
    left = x0 + 1
    bottom = top + inner_h - 1

    bins = downsample_to_width(norm_bins, inner_w)

    # Bars
    for x in range(inner_w):
        v = float(bins[x])
        bar = int(v * (inner_h - 1))
        for k in range(bar):
            y = bottom - k
            # stronger values: more solid char
            val01 = (k + 1) / max(1, inner_h - 1)
            _, attr = val_to_style(max(v, val01), pair_ids)
            safe_addch(stdscr, y, left + x, BLOCK, attr)

    # Mark tone frequencies with a caret on the top row
    nyq = sr / 2.0
    for f in tone_freqs:
        if 0.0 <= f <= nyq:
            xx = left + int((f / nyq) * (inner_w - 1))
            safe_addch(stdscr, top, xx, "^", curses.A_BOLD)

    # Axis labels
    safe_addstr(stdscr, y0 + h - 1, x0 + 2, "0Hz")
    label = f"{int(nyq)}Hz"
    safe_addstr(stdscr, y0 + h - 1, x0 + w - 2 - len(label), label)


def draw_spectrogram(stdscr, y0: int, x0: int, h: int, w: int,
                     spec_hist: np.ndarray, pair_ids: List[int]):
    if h < 3 or w < 5:
        return
    draw_box(stdscr, y0, x0, h, w, "Spectrogram (waterfall)")

    inner_h = h - 2
    inner_w = w - 2
    top = y0 + 1
    left = x0 + 1

    # spec_hist shape: (inner_h, inner_w)
    if spec_hist.shape != (inner_h, inner_w):
        return

    for row in range(inner_h):
        y = top + row
        for col in range(inner_w):
            v = float(spec_hist[row, col])
            ch, attr = val_to_style(v, pair_ids)
            safe_addch(stdscr, y, left + col, ch, attr)


HELP_TEXT = [
    "Controls:",
    "  q            quit",
    "  space        pause/resume",
    "  h            toggle help",
    "  1/2/3        select tone",
    "  \u2190/\u2192          selected tone frequency down/up",
    "  \u2193/\u2191          selected tone amplitude down/up",
    "  n            toggle noise",
    "  +/-          noise amplitude down/up",
    "  l            toggle log spectrum (dB) vs linear",
    "  w            toggle Hann window",
    "  r            reset defaults",
]


# ----------------------------
# Main loop
# ----------------------------
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)

    pair_ids, levels = init_color_pairs(n_levels=48)
    if levels < 8:
        # In very limited terminals, still proceed
        pass

    def reset_params():
        return (
            [Tone(7.0, 0.9), Tone(23.0, 0.55), Tone(41.0, 0.35)],
            0,      # selected tone
            0.08,   # noise amp
            True,   # noise on
            True,   # log scale
            True,   # window on
        )

    tones, sel, noise_amp, noise_on, log_scale, use_window = reset_params()

    sr = 400.0
    n_fft = 512
    fps = 28.0
    step = max(1, int(sr / fps))

    buf = np.zeros(n_fft, dtype=np.float32)
    t_now = 0.0

    # Prime buffer
    x, t_now = gen_samples(t_now, n_fft, sr, tones, noise_amp if noise_on else 0.0)
    buf[:] = x

    w_hann = hann_window(n_fft)

    paused = False
    show_help = False

    mag_ema = None
    ema_a = 0.65  # higher = smoother (more lag)

    last_h, last_w = stdscr.getmaxyx()
    spec_hist = None  # allocated when layout is known

    while True:
        # Handle input
        ch = stdscr.getch()
        while ch != -1:
            if ch in (ord("q"), ord("Q")):
                return
            elif ch == ord(" "):
                paused = not paused
            elif ch in (ord("h"), ord("H")):
                show_help = not show_help
            elif ch in (ord("l"), ord("L")):
                log_scale = not log_scale
            elif ch in (ord("w"), ord("W")):
                use_window = not use_window
            elif ch in (ord("n"), ord("N")):
                noise_on = not noise_on
            elif ch == ord("+"):
                noise_amp = min(1.0, noise_amp + 0.02)
            elif ch == ord("-"):
                noise_amp = max(0.0, noise_amp - 0.02)
            elif ch in (ord("r"), ord("R")):
                tones, sel, noise_amp, noise_on, log_scale, use_window = reset_params()
                mag_ema = None
                t_now = 0.0
                x, t_now = gen_samples(t_now, n_fft, sr, tones, noise_amp if noise_on else 0.0)
                buf[:] = x
            elif ch in (ord("1"), ord("2"), ord("3")):
                sel = int(chr(ch)) - 1
                sel = max(0, min(sel, len(tones) - 1))
            elif ch == curses.KEY_LEFT:
                tones[sel].freq = max(0.2, tones[sel].freq - 0.5)
            elif ch == curses.KEY_RIGHT:
                tones[sel].freq = min(sr / 2.0 - 1.0, tones[sel].freq + 0.5)
            elif ch == curses.KEY_DOWN:
                tones[sel].amp = max(0.0, tones[sel].amp - 0.03)
            elif ch == curses.KEY_UP:
                tones[sel].amp = min(2.0, tones[sel].amp + 0.03)

            ch = stdscr.getch()

        # Resize detection
        h, w = stdscr.getmaxyx()
        if (h, w) != (last_h, last_w):
            last_h, last_w = h, w
            stdscr.erase()
            spec_hist = None  # reallocate

        # Layout
        header_h = 3
        footer_h = 1

        usable_h = max(0, h - header_h - footer_h)
        wave_h = max(8, int(usable_h * 0.33))
        spec_h = max(7, int(usable_h * 0.27))
        specg_h = max(6, usable_h - wave_h - spec_h)

        # Ensure panels fit
        if header_h + wave_h + spec_h + specg_h + footer_h > h:
            specg_h = max(3, h - header_h - wave_h - spec_h - footer_h)

        # Update signal + FFT
        if not paused:
            new, t_now = gen_samples(t_now, step, sr, tones, noise_amp if noise_on else 0.0)
            if step >= n_fft:
                buf[:] = new[-n_fft:]
            else:
                buf[:-step] = buf[step:]
                buf[-step:] = new

        xw = buf * (w_hann if use_window else 1.0)
        spec = np.fft.rfft(xw)
        mag = np.abs(spec).astype(np.float32)

        if mag_ema is None:
            mag_ema = mag
        else:
            mag_ema = ema_a * mag_ema + (1.0 - ema_a) * mag

        norm = mag_to_norm(mag_ema, log_scale=log_scale)

        # Allocate / update spectrogram history
        inner_w = max(1, w - 2)
        inner_h = max(1, specg_h - 2)
        if inner_w < 5 or inner_h < 3:
            spec_hist = None
        else:
            row = downsample_to_width(norm, inner_w)
            if spec_hist is None or spec_hist.shape != (inner_h, inner_w):
                spec_hist = np.zeros((inner_h, inner_w), dtype=np.float32)
            spec_hist[1:] = spec_hist[:-1]
            spec_hist[0] = row

        # Draw
        stdscr.erase()

        # Header
        title = "Live FFT Visualizer (curses + ANSI gradient)"
        safe_addstr(stdscr, 0, max(0, (w - len(title)) // 2), title, curses.A_BOLD)

        tone_str = " | ".join(
            [f"{i+1}:{'>' if i==sel else ' '} f={t.freq:5.1f}Hz a={t.amp:4.2f}"
             for i, t in enumerate(tones)]
        )
        status = f"sr={int(sr)}Hz  N={n_fft}  step={step}  {'LOG(dB)' if log_scale else 'LIN'}  {'Hann' if use_window else 'Rect'}  noise={'ON' if noise_on else 'OFF'}({noise_amp:.2f})  {'PAUSED' if paused else ''}"
        safe_addstr(stdscr, 1, 2, tone_str[: max(0, w - 4)])
        safe_addstr(stdscr, 2, 2, status[: max(0, w - 4)])

        # Panels
        y = header_h
        draw_waveform(stdscr, y, 0, wave_h, w, buf, pair_ids)
        y += wave_h
        tone_freqs = [t.freq for t in tones]
        draw_spectrum(stdscr, y, 0, spec_h, w, norm, pair_ids, tone_freqs, sr)
        y += spec_h
        if spec_hist is not None:
            draw_spectrogram(stdscr, y, 0, specg_h, w, spec_hist, pair_ids)
        else:
            draw_box(stdscr, y, 0, max(3, specg_h), w, "Spectrogram")
            safe_addstr(stdscr, y + 1, 2, "Terminal too small for spectrogram.")

        # Footer
        footer = "q quit | space pause | h help | 1/2/3 select | arrows tweak | n noise | +/- noise level | l log/lin | w window | r reset"
        safe_addstr(stdscr, h - 1, 0, footer[:w - 1])

        # Help overlay
        if show_help:
            box_w = min(w - 4, 54)
            box_h = min(h - 4, len(HELP_TEXT) + 2)
            by = max(1, (h - box_h) // 2)
            bx = max(1, (w - box_w) // 2)
            draw_box(stdscr, by, bx, box_h, box_w, "Help")
            for i, line in enumerate(HELP_TEXT[: box_h - 2]):
                safe_addstr(stdscr, by + 1 + i, bx + 2, line[: box_w - 4])

        stdscr.refresh()

        # Frame limiting
        time.sleep(1.0 / fps)


if __name__ == "__main__":
    curses.wrapper(main)
