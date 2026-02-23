#!/usr/bin/env python3
import curses
import json
import time
import random
import argparse
import ast
import locale
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from bisect import bisect_right


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Step:
    i: int
    pc: str
    bytes: str
    asm: str
    regs_w: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # reg -> (old, new)
    mem_w: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # addr -> (old, new)
    taken: Optional[bool] = None
    target: Optional[str] = None


def _norm_pairs(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, list) and len(v) == 2:
            out[str(k)] = (str(v[0]), str(v[1]))
        elif isinstance(v, tuple) and len(v) == 2:
            out[str(k)] = (str(v[0]), str(v[1]))
        else:
            out[str(k)] = ("?", str(v))
    return out


def parse_obj_line(line: str, line_no: int, path: str) -> dict:
    """
    Accept:
      - JSON objects (double quotes)
      - Python dict literals (single quotes) via ast.literal_eval
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(line)
            if not isinstance(obj, dict):
                raise ValueError("line is not a dict/object")
            return obj
        except Exception as e:
            raise ValueError(
                f"{path}:{line_no}: could not parse line as JSON or Python dict literal.\n"
                f"Line: {line[:200]}"
            ) from e


def load_trace_jsonl(path: str) -> List[Step]:
    steps: List[Step] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue

            obj = parse_obj_line(line, line_no, path)

            step = Step(
                i=int(obj.get("i", len(steps))),
                pc=str(obj.get("pc", "?")),
                bytes=str(obj.get("bytes", "")),
                asm=str(obj.get("asm", "")),
                regs_w=_norm_pairs(obj.get("regs_w")),
                mem_w=_norm_pairs(obj.get("mem_w")),
                taken=obj.get("taken", None),
                target=obj.get("target", None),
            )
            steps.append(step)

    # Ensure indices are monotonic
    for idx, s in enumerate(steps):
        s.i = idx
    return steps


def make_demo_trace(n: int = 600) -> List[Step]:
    pcs = [0x401000 + 5 * i for i in range(n)]
    regs = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RSP", "RBP", "R8", "R9"]
    mnems = [
        "push rbp",
        "mov rbp, rsp",
        "sub rsp, 0x20",
        "mov rax, rbx",
        "add rax, 1",
        "mov [rbp-0x8], rax",
        "mov rax, [rbp-0x8]",
        "cmp rax, 0",
        "jne 0x4010A0",
        "call 0x401200",
        "leave",
        "ret",
    ]
    steps: List[Step] = []
    regvals = {r: random.randint(0, 2**16) for r in regs}
    regvals["RSP"] = 0x7FFFFFFFE000
    regvals["RBP"] = 0x0

    mem = {}  # addr(int)-> value(int)
    for i in range(n):
        pc = pcs[i]
        asm = random.choice(mnems)
        b = f"{random.randint(0,255):02x} " * random.randint(1, 6)

        regs_w = {}
        mem_w = {}
        taken = None
        target = None

        def wreg(r, newv):
            old = regvals.get(r, 0)
            regvals[r] = newv & 0xFFFFFFFFFFFFFFFF
            regs_w[r] = (hex(old), hex(regvals[r]))

        def wmem(addr, newv):
            old = mem.get(addr, 0)
            mem[addr] = newv & 0xFFFFFFFFFFFFFFFF
            mem_w[hex(addr)] = (hex(old), hex(mem[addr]))

        # crude semantics to make stack visuals interesting
        if asm == "push rbp":
            wreg("RSP", regvals["RSP"] - 8)
            wmem(regvals["RSP"], regvals["RBP"])
        elif asm == "mov rbp, rsp":
            wreg("RBP", regvals["RSP"])
        elif asm.startswith("sub rsp"):
            wreg("RSP", regvals["RSP"] - 0x20)
        elif asm.startswith("mov [rbp-0x8]"):
            addr = regvals["RBP"] - 8
            wmem(addr, regvals.get("RAX", 0))
        elif asm.startswith("mov rax, [rbp-0x8]"):
            addr = regvals["RBP"] - 8
            wreg("RAX", mem.get(addr, 0))
        elif asm == "leave":
            # mov rsp, rbp; pop rbp
            wreg("RSP", regvals["RBP"])
            new_rbp = mem.get(regvals["RSP"], 0)
            wreg("RBP", new_rbp)
            wreg("RSP", regvals["RSP"] + 8)
        elif asm == "ret":
            wreg("RSP", regvals["RSP"] + 8)
        elif asm.startswith("jne"):
            taken = random.random() < 0.4
            target = asm.split()[-1]
        else:
            if random.random() < 0.6:
                r = random.choice([x for x in regs if x not in ("RSP", "RBP")])
                old = regvals.get(r, 0)
                new = (old + random.randint(-50, 200)) & 0xFFFFFFFFFFFFFFFF
                wreg(r, new)

        steps.append(
            Step(
                i=i,
                pc=hex(pc),
                bytes=b.strip(),
                asm=asm,
                regs_w=regs_w,
                mem_w=mem_w,
                taken=taken,
                target=target,
            )
        )
    return steps


# -----------------------------
# State snapshots for fast seeking
# -----------------------------
def parse_hex_int(s: str) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if not s:
        return None
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s, 10)
    except Exception:
        return None


@dataclass
class Snap:
    reg: Dict[str, str]
    mem: Dict[int, str]            # addr(int) -> value(str)
    mem_last_w: Dict[int, int]     # addr(int) -> step index last written


def build_snaps(steps: List[Step], stride: int = 256) -> Tuple[List[int], Dict[int, Snap]]:
    reg_state: Dict[str, str] = {}
    mem_state: Dict[int, str] = {}
    mem_last_w: Dict[int, int] = {}
    snaps: Dict[int, Snap] = {}
    snap_keys: List[int] = []

    snaps[0] = Snap(reg_state.copy(), mem_state.copy(), mem_last_w.copy())
    snap_keys.append(0)

    for i, st in enumerate(steps):
        for r, (_old, new) in st.regs_w.items():
            reg_state[r] = new
        for a, (_old, new) in st.mem_w.items():
            ai = parse_hex_int(a)
            if ai is None:
                continue
            mem_state[ai] = new
            mem_last_w[ai] = i

        if (i + 1) % stride == 0:
            k = i
            snaps[k] = Snap(reg_state.copy(), mem_state.copy(), mem_last_w.copy())
            snap_keys.append(k)

    if snap_keys[-1] != len(steps) - 1:
        k = len(steps) - 1
        snaps[k] = Snap(reg_state.copy(), mem_state.copy(), mem_last_w.copy())
        snap_keys.append(k)

    snap_keys = sorted(set(snap_keys))
    return snap_keys, snaps


def state_at(idx: int, steps: List[Step], snap_keys: List[int], snaps: Dict[int, Snap]) -> Snap:
    if idx <= 0:
        base_k = 0
    else:
        j = bisect_right(snap_keys, idx) - 1
        base_k = snap_keys[max(0, j)]

    base = snaps[base_k]
    reg = base.reg.copy()
    mem = base.mem.copy()
    mem_last_w = base.mem_last_w.copy()

    start = 0 if base_k == 0 else base_k + 1
    for i in range(start, idx + 1):
        st = steps[i]
        for r, (_old, new) in st.regs_w.items():
            reg[r] = new
        for a, (_old, new) in st.mem_w.items():
            ai = parse_hex_int(a)
            if ai is None:
                continue
            mem[ai] = new
            mem_last_w[ai] = i

    return Snap(reg, mem, mem_last_w)


# -----------------------------
# Curses safe drawing + colors
# -----------------------------
def safe_addnstr(win, y: int, x: int, s: str, n: int, attr: int = 0):
    """
    Draw without ever touching the bottom-right cell (a common ERR source).
    Also clamps to window size and ignores resize-time ERRs.
    """
    try:
        h, w = win.getmaxyx()
        if h <= 0 or w <= 0:
            return
        if y < 0 or y >= h:
            return
        if x < 0 or x >= w:
            return
        # never write into last column
        max_n = max(0, (w - 1) - x)
        if max_n <= 0:
            return
        n = min(n, max_n)
        if n <= 0:
            return
        win.addnstr(y, x, s, n, attr)
    except curses.error:
        pass


def init_colors():
    curses.start_color()
    try:
        curses.use_default_colors()
    except Exception:
        pass

    max_colors = getattr(curses, "COLORS", 0)
    max_pairs = getattr(curses, "COLOR_PAIRS", 0)

    if max_colors >= 256:
        ramp = list(range(236, 256))  # grayscale ramp
    elif max_colors >= 16:
        ramp = [curses.COLOR_BLACK, curses.COLOR_BLUE, curses.COLOR_CYAN, curses.COLOR_WHITE]
    else:
        ramp = [curses.COLOR_WHITE]

    pairs = []
    pid = 1
    for c in ramp:
        if pid >= max_pairs:
            break
        try:
            curses.init_pair(pid, c, -1)
        except Exception:
            curses.init_pair(pid, c, curses.COLOR_BLACK)
        pairs.append(pid)
        pid += 1

    if not pairs:
        pairs = [0]
    return pairs


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def heat_pair(pairs, dist: int) -> int:
    n = len(pairs)
    d = clamp(dist, 0, n - 1)
    return pairs[n - 1 - d]


def draw_bar(stdscr, y, text, attr=0):
    h, w = stdscr.getmaxyx()
    if w <= 2 or h <= 0:
        return
    safe_addnstr(stdscr, y, 0, text.ljust(w - 1), w - 1, attr)


def fmt_row(step: Step, w: int) -> str:
    idx = f"{step.i:6d}"
    pc = f"{step.pc:>10}"
    byt = step.bytes.replace("  ", " ")[:16].ljust(16)
    asm = step.asm
    s = f"{idx}  {pc}  {byt}  {asm}"
    return s


def draw_effects(win, step: Step):
    win.erase()
    h, w = win.getmaxyx()
    y = 0

    def add(line, attr=0):
        nonlocal y
        if y >= h:
            return
        safe_addnstr(win, y, 0, line.ljust(w - 1), w - 1, attr)
        y += 1

    add("Effects", curses.A_BOLD)
    add("-" * min(w - 1, 40))

    if step.taken is not None:
        add(f"Branch: {'TAKEN' if step.taken else 'not taken'}")
        if step.target:
            add(f"Target: {step.target}")
        add("")

    if step.regs_w:
        add("Regs written:", curses.A_BOLD)
        for r, (old, new) in list(step.regs_w.items())[: max(0, h - y - 1)]:
            add(f" {r:>4}: {old} -> {new}")
        add("")
    else:
        add("Regs written: (none)")
        add("")

    if step.mem_w and y < h:
        add("Mem written:", curses.A_BOLD)
        for a, (old, new) in list(step.mem_w.items())[: max(0, h - y - 1)]:
            add(f" {a}: {old} -> {new}")

    win.noutrefresh()


def draw_regs(win, reg: Dict[str, str], regs_w_this_step: Dict[str, Tuple[str, str]], pairs):
    win.erase()
    h, w = win.getmaxyx()
    y = 0

    def add(line, attr=0):
        nonlocal y
        if y >= h:
            return
        safe_addnstr(win, y, 0, line.ljust(w - 1), w - 1, attr)
        y += 1

    add("Registers", curses.A_BOLD)
    add("-" * min(w - 1, 40))

    prefer = ["RIP", "RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RSP", "RBP",
              "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15"]
    shown = [r for r in prefer if r in reg]
    if not shown:
        shown = sorted(reg.keys())

    max_items = max(0, h - y)
    for r in shown[:max_items]:
        v = reg.get(r, "?")
        attr = 0
        if r in regs_w_this_step:
            attr |= curses.A_BOLD | curses.color_pair(heat_pair(pairs, 0))
        add(f" {r:>4} = {v}", attr)

    win.noutrefresh()


def draw_stack_list(win, idx: int, reg: Dict[str, str], mem: Dict[int, str], mem_last_w: Dict[int, int], pairs):
    win.erase()
    h, w = win.getmaxyx()
    y = 0

    def add(line, attr=0):
        nonlocal y
        if y >= h:
            return
        safe_addnstr(win, y, 0, line.ljust(w - 1), w - 1, attr)
        y += 1

    add("Stack (around RSP)", curses.A_BOLD)
    add("-" * min(w - 1, 40))

    rsp = parse_hex_int(reg.get("RSP", ""))
    if rsp is None:
        add("RSP: (unknown)")
        win.noutrefresh()
        return

    words_above = max(1, (h - y - 1) // 2)
    words_below = (h - y - 1) - words_above
    start = rsp - words_above * 8

    for k in range(words_above + 1 + words_below):
        addr = start + k * 8
        val = mem.get(addr, "........")
        lastw = mem_last_w.get(addr, -10**9)
        dist = idx - lastw
        pid = heat_pair(pairs, dist) if lastw > -10**8 else pairs[0]
        attr = curses.color_pair(pid)

        marker = "->" if addr == rsp else "  "
        sval = str(val)
        if sval.startswith("0x") and len(sval) > 10:
            sval = "0x.." + sval[-8:]  # ASCII-only

        add(
            f"{marker} {hex(addr):>14}  {sval:>12}",
            (curses.A_BOLD | attr) if addr == rsp else attr
        )

    win.noutrefresh()


def draw_stack_grid(win, idx: int, reg: Dict[str, str], mem: Dict[int, str], mem_last_w: Dict[int, int], pairs):
    win.erase()
    h, w = win.getmaxyx()

    rsp = parse_hex_int(reg.get("RSP", ""))
    title = "Stack Grid (~0x100 bytes around RSP)" if rsp is not None else "Stack Grid"
    safe_addnstr(win, 0, 0, title.ljust(w - 1), w - 1, curses.A_BOLD)
    if h < 4:
        win.noutrefresh()
        return

    if rsp is None:
        safe_addnstr(win, 2, 0, "RSP: (unknown)".ljust(w - 1), w - 1, 0)
        win.noutrefresh()
        return

    base = rsp & ~0xFF
    rows, cols = 8, 8

    cell_w = 6  # "abcd "
    addr_w = 14
    needed_w = addr_w + cols * cell_w + 1
    if needed_w > w:
        cols = max(4, (w - addr_w - 1) // cell_w)

    start_y = 2
    header = " " * addr_w
    for c in range(cols):
        header += f"{c:02X} ".rjust(cell_w)
    safe_addnstr(win, start_y, 0, header.ljust(w - 1), w - 1, curses.A_DIM)

    max_rows = min(rows, h - (start_y + 1) - 1)
    for r in range(max_rows):
        y = start_y + 1 + r
        row_addr = base + r * (cols * 8)

        safe_addnstr(win, y, 0, f"{hex(row_addr):>14} ", min(w - 1, addr_w), curses.A_DIM)

        for c in range(cols):
            addr = row_addr + c * 8
            val = mem.get(addr, None)
            lastw = mem_last_w.get(addr, -10**9)
            dist = idx - lastw

            pid = heat_pair(pairs, dist) if val is not None else pairs[0]
            attr = curses.color_pair(pid)
            if addr == rsp:
                attr |= curses.A_BOLD | curses.A_REVERSE

            snippet = ".... "  # ASCII-only
            if val is not None:
                s = str(val)
                if s.startswith("0x"):
                    s = s[2:]
                snippet = (s[-4:] if len(s) >= 4 else s.rjust(4, "0")) + " "

            safe_addnstr(win, y, addr_w + c * cell_w, snippet[:cell_w], cell_w, attr)

    legend = "Heat=recent writes (bright). [RSP] highlighted."
    safe_addnstr(win, h - 1, 0, legend.ljust(w - 1), w - 1, curses.A_DIM)
    win.noutrefresh()


# -----------------------------
# Main loop
# -----------------------------
def main_curses(stdscr, steps: List[Step], start_paused: bool = False):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    pairs = init_colors()
    grad_n = len(pairs)

    snap_keys, snaps = build_snaps(steps, stride=256)

    idx = 0
    paused = start_paused
    speed_hz = 15.0
    last_tick = time.time()

    show_inspector = True
    show_stack_grid_mode = True  # True=grid (and list if tall), False=list-only
    scroll_offset = 0

    cur_snap = state_at(0, steps, snap_keys, snaps)

    def refresh_state_to(new_idx: int):
        nonlocal idx, cur_snap
        if new_idx == idx:
            return
        idx = new_idx
        cur_snap = state_at(idx, steps, snap_keys, snaps)

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        if h < 4 or w < 20:
            draw_bar(stdscr, 0, "Terminal too small. Resize bigger. (q to quit)", curses.A_REVERSE)
            draw_bar(stdscr, h - 1, "q quit", curses.A_REVERSE)
            stdscr.noutrefresh()
            curses.doupdate()
            try:
                key = stdscr.getch()
            except Exception:
                key = -1
            if key in (ord("q"), 27):
                break
            time.sleep(0.05)
            continue

        inspector_w = max(0, min(52, w // 2)) if show_inspector else 0
        river_w = w - inspector_w
        river_h = h - 2
        river_y0 = 1

        cur_step = steps[idx]

        status = (
            f" Trace River | step {idx+1}/{len(steps)} | PC {cur_step.pc} | "
            f"{'PAUSED' if paused else f'RUN {speed_hz:.1f}x'} | inspector {'ON' if show_inspector else 'OFF'} "
        )
        draw_bar(stdscr, 0, status, curses.A_REVERSE)

        # Inspector
        if show_inspector and inspector_w >= 24:
            x0 = river_w
            avail_h = river_h

            eff_h = min(8, max(6, avail_h // 4))
            reg_h = min(8, max(6, avail_h // 4))
            remain = avail_h - eff_h - reg_h

            if show_stack_grid_mode:
                show_both = remain >= 14
                if show_both:
                    stack_list_h = remain // 2
                    stack_grid_h = remain - stack_list_h
                else:
                    stack_list_h = 0
                    stack_grid_h = max(6, remain)
            else:
                stack_list_h = max(6, remain)
                stack_grid_h = 0

            y = river_y0
            eff_win = stdscr.derwin(eff_h, inspector_w, y, x0)
            draw_effects(eff_win, cur_step)
            y += eff_h

            reg_win = stdscr.derwin(reg_h, inspector_w, y, x0)
            draw_regs(reg_win, cur_snap.reg, cur_step.regs_w, pairs)
            y += reg_h

            if stack_list_h > 0:
                sl_win = stdscr.derwin(stack_list_h, inspector_w, y, x0)
                draw_stack_list(sl_win, idx, cur_snap.reg, cur_snap.mem, cur_snap.mem_last_w, pairs)
                y += stack_list_h

            if stack_grid_h > 0:
                sg_h = max(1, avail_h - (y - river_y0))
                sg_win = stdscr.derwin(sg_h, inspector_w, y, x0)
                draw_stack_grid(sg_win, idx, cur_snap.reg, cur_snap.mem, cur_snap.mem_last_w, pairs)

        # River
        center = clamp(idx + scroll_offset, 0, len(steps) - 1)
        half = river_h // 2
        start = clamp(center - half, 0, max(0, len(steps) - river_h))
        end = min(len(steps), start + river_h)

        for row, si in enumerate(range(start, end)):
            step = steps[si]
            dist = abs(si - idx)
            t = clamp(dist, 0, grad_n - 1)
            pid = pairs[max(0, grad_n - 1 - t)]
            attr = curses.color_pair(pid)
            if si == idx:
                attr |= curses.A_BOLD

            line = fmt_row(step, river_w)
            safe_addnstr(stdscr, river_y0 + row, 0, line, river_w - 1, attr)

        help_line = (
            " q quit | space pause | n/p step | j/k scroll (paused) | +/- speed | "
            "t inspector | m grid<->list | g/G start/end "
        )
        draw_bar(stdscr, h - 1, help_line, curses.A_REVERSE)

        stdscr.noutrefresh()
        curses.doupdate()

        # Input
        try:
            key = stdscr.getch()
        except Exception:
            key = -1

        if key != -1:
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
                scroll_offset = 0
            elif key == ord("t"):
                show_inspector = not show_inspector
            elif key == ord("m"):
                show_stack_grid_mode = not show_stack_grid_mode
            elif key in (ord("+"), ord("=")):
                speed_hz = min(240.0, speed_hz * 1.25)
            elif key in (ord("-"), ord("_")):
                speed_hz = max(0.5, speed_hz / 1.25)
            elif key == ord("g"):
                refresh_state_to(0)
                scroll_offset = 0
            elif key == ord("G"):
                refresh_state_to(len(steps) - 1)
                scroll_offset = 0
            elif key == ord("n"):
                paused = True
                refresh_state_to(min(len(steps) - 1, idx + 1))
                scroll_offset = 0
            elif key == ord("p"):
                paused = True
                refresh_state_to(max(0, idx - 1))
                scroll_offset = 0
            elif key == ord("j") and paused:
                scroll_offset = clamp(scroll_offset + 1, -idx, (len(steps) - 1 - idx))
            elif key == ord("k") and paused:
                scroll_offset = clamp(scroll_offset - 1, -idx, (len(steps) - 1 - idx))

        # Playback
        now = time.time()
        if not paused:
            dt = now - last_tick
            step_advance = int(dt * speed_hz)
            if step_advance > 0:
                refresh_state_to(min(len(steps) - 1, idx + step_advance))
                last_tick = now
                if idx >= len(steps) - 1:
                    paused = True
        else:
            last_tick = now

        time.sleep(0.01)


def main():
    # Helps curses behave with your terminal locale; we still keep UI ASCII-safe.
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="ASCII Trace River + Stack/Memory Grid (curses)")
    ap.add_argument("trace", nargs="?", default=None, help="Path to trace.jsonl (optional: runs demo if omitted)")
    ap.add_argument("--paused", action="store_true", help="Start paused")
    args = ap.parse_args()

    steps = load_trace_jsonl(args.trace) if args.trace else make_demo_trace()
    if not steps:
        print("No steps to display.")
        return

    curses.wrapper(main_curses, steps, args.paused)


if __name__ == "__main__":
    main()
