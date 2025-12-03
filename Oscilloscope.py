// Author:  William Selby
#!/usr/bin/env python3
"""
oscilloscope_cli.py — rolling ASCII oscilloscope with 1D (time) and 2D (X–Y) modes
+ simulators for Mass–Spring–Damper, Lorenz, Double Pendulum, and Euler (Cornu) spiral.

Quick examples
--------------
# Lissajous (X–Y expressions)
./oscilloscope_cli.py --xy \
  --xexpr "sin(2*pi*3*t)" \
  --yexpr "sin(2*pi*2*t)" \
  --fps 60

# MSD phase portrait (x vs v)
./oscilloscope_cli.py --msd --xy --msd-phase \
  --m 1 --c 0.2 --k 25 --force-expr "(t<0.2)*10"

# Lorenz projection (x vs z)
./oscilloscope_cli.py --lorenz --xy --xy-components xz --fps 60

# Double pendulum — end-effector XY trail
./oscilloscope_cli.py --pendulum --xy --t1 1.2 --t2 2.3 --fps 60

# Euler (Cornu) spiral — smooth XY trail
./oscilloscope_cli.py --euler --xy --euler-speed 0.8 --fps 60

Keys
----
q quit • space pause/resume • a autoscale • g grid • c color • r reset
Lorenz: (1/2/3) switch component/pair (1D x/y/z; XY xy/xz/yz)
"""

import argparse
import curses
import time
import math
import ast
from collections import deque
from typing import Optional, Tuple

# ---------------------- Safe expression evaluator ---------------------- #
_ALLOWED_NAMES = {
    'pi': math.pi, 'e': math.e,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'exp': math.exp, 'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt,
    'floor': math.floor, 'ceil': math.ceil, 'fabs': math.fabs, 'abs': abs, 'pow': pow,
}

class SafeExpr:
    def __init__(self, expr: Optional[str]):
        self.expr = expr or "0"
        try:
            self._tree = ast.parse(self.expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        self._validate(self._tree)

    def _validate(self, node):
        if isinstance(node, ast.Expression):
            self._validate(node.body)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants allowed")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                raise ValueError("Unsupported binary operator")
            self._validate(node.left); self._validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError("Unsupported unary operator")
            self._validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError(f"Function '{getattr(node.func, 'id', '?')}' not allowed")
            for arg in node.args: self._validate(arg)
            if getattr(node, 'keywords', None):
                if any(k.arg is not None for k in node.keywords):
                    raise ValueError("Keyword arguments not allowed")
        elif isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES and node.id != 't':
                raise ValueError(f"Name '{node.id}' not allowed; only 't' and math funcs")
        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def eval(self, t: float) -> float:
        env = dict(_ALLOWED_NAMES); env['t'] = t
        return self._eval_node(self._tree.body, env)

    def _eval_node(self, node, env):
        if isinstance(node, ast.Constant): return float(node.value)
        if isinstance(node, ast.BinOp):
            l = self._eval_node(node.left, env); r = self._eval_node(node.right, env)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.Pow): return l ** r
            if isinstance(node.op, ast.Mod): return l % r
        if isinstance(node, ast.UnaryOp):
            v = self._eval_node(node.operand, env)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.USub): return -v
        if isinstance(node, ast.Call):
            fn = _ALLOWED_NAMES[node.func.id]
            args = [self._eval_node(a, env) for a in node.args]
            return float(fn(*args))
        if isinstance(node, ast.Name): return float(env[node.id])
        raise RuntimeError(f"Cannot evaluate node type {type(node).__name__}")

# --------------------------- Signal sources ---------------------------- #
class ExprSource1D:
    def __init__(self, expr: SafeExpr):
        self.expr = expr; self.t0 = time.perf_counter(); self.t = 0.0
    def reset(self): self.t0 = time.perf_counter(); self.t = 0.0
    def step(self, dt: float) -> Tuple[float, float]:
        self.t = time.perf_counter() - self.t0
        return self.t, float(self.expr.eval(self.t))

class ExprSourceXY:
    def __init__(self, xexpr: SafeExpr, yexpr: SafeExpr):
        self.xexpr, self.yexpr = xexpr, yexpr
        self.t0 = time.perf_counter(); self.t = 0.0
    def reset(self): self.t0 = time.perf_counter(); self.t = 0.0
    def step(self, dt: float) -> Tuple[float, Tuple[float, float]]:
        self.t = time.perf_counter() - self.t0
        return self.t, (float(self.xexpr.eval(self.t)), float(self.yexpr.eval(self.t)))

class MSDSource1D:
    """m*x'' + c*x' + k*x = F(t) — RK4 integration; outputs x(t)."""
    def __init__(self, m=1.0, c=0.2, k=10.0, x0=0.0, v0=0.0, force_expr: Optional[SafeExpr]=None):
        self.m=float(m); self.c=float(c); self.k=float(k)
        self.x=float(x0); self.v=float(v0)
        self.xi=float(x0); self.vi=float(v0)
        self.force = force_expr or SafeExpr("0")
        self.t0 = time.perf_counter(); self.t = 0.0
    def reset(self): self.x=self.xi; self.v=self.vi; self.t0=time.perf_counter(); self.t=0.0
    def _accel(self, t, x, v): return (float(self.force.eval(t)) - self.c*v - self.k*x)/self.m
    def _rk4(self, t, h):
        x, v = self.x, self.v
        a1 = self._accel(t, x, v); k1x, k1v = v, a1
        a2 = self._accel(t+0.5*h, x+0.5*h*k1x, v+0.5*h*k1v); k2x, k2v = v+0.5*h*k1v, a2
        a3 = self._accel(t+0.5*h, x+0.5*h*k2x, v+0.5*h*k2v); k3x, k3v = v+0.5*h*k2v, a3
        a4 = self._accel(t+h,   x+h*k3x,       v+h*k3v);     k4x, k4v = v+h*k3v, a4
        self.x = x + (h/6.0)*(k1x+2*k2x+2*k3x+k4x)
        self.v = v + (h/6.0)*(k1v+2*k2v+2*k3v+k4v)
    def step(self, dt: float) -> Tuple[float, float]:
        if dt <= 0: dt = 1e-3
        max_step = 1/600.0; n = max(1, int(dt / max_step)); h = dt / n
        self.t = time.perf_counter() - self.t0
        for _ in range(n): self._rk4(self.t, h); self.t += h
        return self.t, self.x

class MSDSourceXY:
    def __init__(self, base: MSDSource1D): self.base = base
    def reset(self): self.base.reset()
    def step(self, dt: float) -> Tuple[float, Tuple[float, float]]:
        t, _ = self.base.step(dt); return t, (self.base.x, self.base.v)

class LorenzSource1D:
    """Lorenz system."""
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, x0=1.0, y0=1.0, z0=1.0, component='x'):
        self.sigma=float(sigma); self.rho=float(rho); self.beta=float(beta)
        self.x=float(x0); self.y=float(y0); self.z=float(z0)
        self.xi,self.yi,self.zi=self.x,self.y,self.z
        self.component=component.lower(); self.t=0.0
    def reset(self): self.x,self.y,self.z=self.xi,self.yi,self.zi; self.t=0.0
    def _deriv(self, x, y, z): return self.sigma*(y-x), x*(self.rho - z) - y, x*y - self.beta*z
    def _rk4(self, h):
        x,y,z=self.x,self.y,self.z
        k1x,k1y,k1z=self._deriv(x,y,z)
        k2x,k2y,k2z=self._deriv(x+0.5*h*k1x, y+0.5*h*k1y, z+0.5*h*k1z)
        k3x,k3y,k3z=self._deriv(x+0.5*h*k2x, y+0.5*h*k2y, z+0.5*h*k2z)
        k4x,k4y,k4z=self._deriv(x+h*k3x, y+h*k3y, z+h*k3z)
        self.x = x + (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        self.y = y + (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        self.z = z + (h/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    def step(self, dt: float) -> Tuple[float, float]:
        if dt <= 0: dt = 1e-3
        max_step = 1/2000.0; n = max(1, int(dt / max_step)); h = dt / n
        for _ in range(n): self._rk4(h); self.t += h
        comp = {'x': self.x, 'y': self.y, 'z': self.z}[self.component]
        return self.t, comp

class LorenzSourceXY:
    def __init__(self, base: LorenzSource1D, pair='xy'): self.base=base; self.pair=pair
    def reset(self): self.base.reset()
    def step(self, dt: float) -> Tuple[float, Tuple[float, float]]:
        t,_=self.base.step(dt)
        if self.pair=='xz': return t,(self.base.x,self.base.z)
        if self.pair=='yz': return t,(self.base.y,self.base.z)
        return t,(self.base.x,self.base.y)

# ---------------------- Double Pendulum (chaos) ------------------------ #
class DoublePendulum1D:
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81,
                 t1=math.pi/2, t2=math.pi/2, w1=0.0, w2=0.0, component='t1'):
        self.L1=float(L1); self.L2=float(L2); self.m1=float(m1); self.m2=float(m2); self.g=float(g)
        self.t1=float(t1); self.t2=float(t2); self.w1=float(w1); self.w2=float(w2)
        self.t1i=self.t1; self.t2i=self.t2; self.w1i=self.w1; self.w2i=self.w2
        self.component=component.lower(); self.t=0.0
    def reset(self): self.t1=self.t1i; self.t2=self.t2i; self.w1=self.w1i; self.w2=self.w2i; self.t=0.0
    def _accels(self, t1,t2,w1,w2):
        g=self.g; m1=self.m1; m2=self.m2; L1=self.L1; L2=self.L2; d=t1-t2
        den=2*m1+m2 - m2*math.cos(2*d)
        a1=(-g*(2*m1+m2)*math.sin(t1)-m2*g*math.sin(t1-2*t2)-2*math.sin(d)*m2*(w2*w2*L2 + w1*w1*L1*math.cos(d)))/(L1*den)
        a2=(2*math.sin(d)*(w1*w1*L1*(m1+m2) + g*(m1+m2)*math.cos(t1) + w2*w2*L2*m2*math.cos(d)))/(L2*den)
        return a1,a2
    def _rk4(self, h):
        def f(y): t1,t2,w1,w2=y; a1,a2=self._accels(t1,t2,w1,w2); return (w1,w2,a1,a2)
        y0=(self.t1,self.t2,self.w1,self.w2)
        k1=f(y0); y1=tuple(y0[i]+0.5*h*k1[i] for i in range(4))
        k2=f(y1); y2=tuple(y0[i]+0.5*h*k2[i] for i in range(4))
        k3=f(y2); y3=tuple(y0[i]+h*k3[i] for i in range(4))
        k4=f(y3)
        self.t1=y0[0]+(h/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        self.t2=y0[1]+(h/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        self.w1=y0[2]+(h/6.0)*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        self.w2=y0[3]+(h/6.0)*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
    def step(self, dt: float) -> Tuple[float, float]:
        if dt <= 0: dt=1e-3
        max_step=1/2000.0; n=max(1,int(dt/max_step)); h=dt/n
        for _ in range(n): self._rk4(h)
        val=self.t1 if self.component=='t1' else self.t2
        return 0.0, val
    def pos(self):
        x1=self.L1*math.sin(self.t1); y1=-self.L1*math.cos(self.t1)
        x2=x1+self.L2*math.sin(self.t2); y2=y1-self.L2*math.cos(self.t2)
        return (x1,y1,x2,y2)

class DoublePendulumXY:
    def __init__(self, base: DoublePendulum1D, which='tip'): self.base=base; self.which=which
    def reset(self): self.base.reset()
    def step(self, dt: float) -> Tuple[float, Tuple[float, float]]:
        _,_=self.base.step(dt)
        x1,y1,x2,y2=self.base.pos()
        return (0.0,(x2,y2)) if self.which=='tip' else (0.0,(x1,y1))

# ---------------------- Euler (Cornu) spiral XY ----------------------- #
class EulerSpiralXY:
    """Euler's (Cornu) spiral parameterized by arc length s.
    x(s)=∫0..s cos(π t²/2) dt, y(s)=∫0..s sin(π t²/2) dt
    Integrate dx/ds=cos(0.5*pi*s*s), dy/ds=sin(0.5*pi*s*s).
    """
    def __init__(self, speed: float = 0.8):
        self.s=0.0; self.x=0.0; self.y=0.0; self.speed=float(speed)
    def reset(self): self.s=0.0; self.x=0.0; self.y=0.0
    @staticmethod
    def _f(s): a=0.5*math.pi*s*s; return math.cos(a), math.sin(a)
    def _rk4_step(self,h):
        k1x,k1y=self._f(self.s)
        k2x,k2y=self._f(self.s+0.5*h)
        k3x,k3y=self._f(self.s+0.5*h)
        k4x,k4y=self._f(self.s+h)
        self.x += (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        self.y += (h/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        self.s += h
    def step(self, dt: float) -> Tuple[float, Tuple[float, float]]:
        if dt <= 0: dt=1e-3
        ds=self.speed*dt; max_h=0.01
        n=max(1,int(abs(ds)/max_h)); h=ds/n
        for _ in range(n): self._rk4_step(h)
        return self.s,(self.x,self.y)

# --------------------------- Color management --------------------------- #
class ColorRamp:
    def __init__(self): self.enabled=False; self.pairs=[]
    def setup(self):
        if not curses.has_colors(): self.enabled=False; return
        curses.start_color()
        try: curses.use_default_colors()
        except Exception: pass
        if curses.COLORS and curses.COLORS>=256:
            ramp=[21,27,33,39,45,51,50,49,48,47,46]
            for i,fg in enumerate(ramp[:curses.COLOR_PAIRS-1], start=1):
                try: curses.init_pair(i, fg, -1); self.pairs.append(curses.color_pair(i))
                except Exception: break
        else:
            for i,fg in enumerate([curses.COLOR_BLUE,curses.COLOR_CYAN,curses.COLOR_GREEN], start=1):
                try: curses.init_pair(i, fg, -1); self.pairs.append(curses.color_pair(i))
                except Exception: break
        self.enabled=len(self.pairs)>=2
    def attr_for_fraction(self, frac: float):
        if not self.enabled: return 0
        frac=0.0 if frac<0 else (1.0 if frac>1 else frac)
        idx=int(round(frac*(len(self.pairs)-1))); idx=max(0,min(len(self.pairs)-1,idx))
        return self.pairs[idx]

# ------------------------------ Scopes --------------------------------- #
def clamp(v,a,b): return max(a,min(b,v))

class Scope1D:
    def __init__(self, stdscr, source, fps=60, autoscale=True, grid=True, color_on=True):
        self.stdscr=stdscr; self.source=source; self.fps=max(1,int(fps))
        self.autoscale=autoscale; self.grid=grid; self.color_on=color_on
        self.values=deque(); self.paused=False; self.ramp=ColorRamp(); self.ramp.setup()
        self.ymin,self.ymax=-1.0,1.0; self._tick=time.perf_counter(); self._t=0.0
    def _resize(self):
        h,w=self.stdscr.getmaxyx(); w=max(10,w)
        while len(self.values)>w: self.values.popleft()
        while len(self.values)<w: self.values.append(0.0)
    def _scale(self):
        if not self.autoscale: return self.ymin,self.ymax
        if not self.values: return -1.0,1.0
        vmin,vmax=min(self.values),max(self.values)
        if vmin==vmax:
            pad=1.0 if vmin==0 else abs(vmin)*0.2; return vmin-pad,vmax+pad
        span=vmax-vmin; pad=0.1*span; return vmin-pad,vmax+pad
    def _draw(self):
        self.stdscr.erase(); h,w=self.stdscr.getmaxyx()
        ymin,ymax=(self._scale() if self.autoscale else (self.ymin,self.ymax))
        if ymax-ymin<1e-9: ymax=ymin+1e-9
        def y_to_row(y): frac=(y-ymin)/(ymax-ymin); return clamp(int(round((1-frac)*(h-2))),0,max(0,h-2))
        if self.grid and h>=4 and w>=20:
            if ymin<0<ymax:
                z=y_to_row(0.0)
                for x in range(w):
                    try: self.stdscr.addch(z,x,'-')
                    except curses.error: pass
            for x in range(0,w,10):
                for r in range(0,h-1,2):
                    try: self.stdscr.addch(r,x,'|')
                    except curses.error: pass
        vmin,vmax=(min(self.values),max(self.values)) if self.values else (0.0,1.0)
        denom=(vmax-vmin) or 1.0
        for x,y in enumerate(self.values):
            row=y_to_row(y); attr=0
            if self.color_on and self.ramp.enabled:
                frac=(y-vmin)/denom; attr=self.ramp.attr_for_fraction(frac)
            try: self.stdscr.addch(row,x,ord('*'),attr)
            except curses.error: pass
        try: self.stdscr.addstr(h-1,0,f"1D  t={self._t:7.3f}s  autoscale={'ON' if self.autoscale else 'OFF'}  grid={'ON' if self.grid else 'OFF'}  fps={self.fps}"[:max(0,w-1)])
        except curses.error: pass
        self.stdscr.refresh()
    def run(self):
        self.stdscr.nodelay(True)
        try: curses.curs_set(0)
        except curses.error: pass
        self._resize(); next_frame=time.perf_counter()
        while True:
            try: key=self.stdscr.getch()
            except curses.error: key=-1
            if key!=-1:
                if key in (ord('q'),ord('Q')): break
                elif key==ord(' '): self.paused=not self.paused
                elif key in (ord('a'),ord('A')): self.autoscale=not self.autoscale
                elif key in (ord('g'),ord('G')): self.grid=not self.grid
                elif key in (ord('c'),ord('C')): self.color_on=not self.color_on
                elif key in (ord('r'),ord('R')):
                    if hasattr(self.source,'reset'): self.source.reset()
                    self.values.clear(); self._resize(); self._t=0.0
                elif key in (ord('1'),ord('2'),ord('3')) and isinstance(self.source,LorenzSource1D):
                    self.source.component={ord('1'):'x',ord('2'):'y',ord('3'):'z'}[key]
            now=time.perf_counter()
            if now<next_frame: time.sleep(max(0,next_frame-now)); continue
            dt_wall=now-getattr(self,'_tick',now); self._tick=now; next_frame=now+1.0/self.fps
            self._resize()
            if not self.paused:
                t_now,y=self.source.step(dt_wall); self._t=t_now
                if not math.isfinite(y): y=0.0
                self.values.append(y)
                h,w=self.stdscr.getmaxyx()
                while len(self.values)>w: self.values.popleft()
            self._draw()

class ScopeXY:
    def __init__(self, stdscr, source, fps=60, autoscale=True, grid=True, color_on=True, trail_factor=2.0):
        self.stdscr=stdscr; self.source=source; self.fps=max(1,int(fps))
        self.autoscale=autoscale; self.grid=grid; self.color_on=color_on
        self.trail=deque(); self.ramp=ColorRamp(); self.ramp.setup()
        self.paused=False; self._tick=time.perf_counter(); self._t=0.0
        self.trail_factor=float(trail_factor); self.xmin=self.ymin=-1.0; self.xmax=self.ymax=1.0
        self.fade_chars="·.:oO@"  # oldest→newest
    def _resize(self):
        h,w=self.stdscr.getmaxyx(); max_len=max(20,int(w*self.trail_factor))
        while len(self.trail)>max_len: self.trail.popleft()
    def _scale(self):
        if not self.autoscale: return self.xmin,self.xmax,self.ymin,self.ymax
        if not self.trail: return -1,1,-1,1
        xs=[p[0] for p in self.trail]; ys=[p[1] for p in self.trail]
        xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
        if xmin==xmax: pad=1.0 if xmin==0 else abs(xmin)*0.2; xmin-=pad; xmax+=pad
        if ymin==ymax: pad=1.0 if ymin==0 else abs(ymin)*0.2; ymin-=pad; ymax+=pad
        dx=xmax-xmin; dy=ymax-ymin
        return xmin-0.1*dx, xmax+0.1*dx, ymin-0.1*dy, ymax+0.1*dy
    def _draw(self):
        self.stdscr.erase(); h,w=self.stdscr.getmaxyx()
        xmin,xmax,ymin,ymax=self._scale()
        if xmax-xmin<1e-12: xmax=xmin+1e-12
        if ymax-ymin<1e-12: ymax=ymin+1e-12
        def x_to_col(x): frac=(x-xmin)/(xmax-xmin); return clamp(int(round(frac*(w-1))),0,max(0,w-1))
        def y_to_row(y): frac=(y-ymin)/(ymax-ymin); return clamp(int(round((1-frac)*(h-2))),0,max(0,h-2))
        def plot(c,r,attr,ch):
            try: self.stdscr.addch(r,c,ord(ch),attr)
            except curses.error: pass
        def line(c0,r0,c1,r1,attr,ch):
            dc=abs(c1-c0); sc=1 if c0<c1 else -1
            dr=-abs(r1-r0); sr=1 if r0<r1 else -1
            err=dc+dr
            while True:
                plot(c0,r0,attr,ch)
                if c0==c1 and r0==r1: break
                e2=2*err
                if e2>=dr: err+=dr; c0+=sc
                if e2<=dc: err+=dc; r0+=sr
        if self.grid and h>=4 and w>=20:
            if xmin<0<xmax:
                x0=x_to_col(0.0)
                for r in range(h-1):
                    try: self.stdscr.addch(r,x0,'|')
                    except curses.error: pass
            if ymin<0<ymax:
                y0=y_to_row(0.0)
                for c in range(w):
                    try: self.stdscr.addch(y0,c,'-')
                    except curses.error: pass
        n=len(self.trail)
        if n<=1:
            if n==1:
                c=x_to_col(self.trail[0][0]); r=y_to_row(self.trail[0][1]); plot(c,r,0,self.fade_chars[-1])
        else:
            for i in range(1,n):
                x0,y0=self.trail[i-1]; x1,y1=self.trail[i]
                c0,r0=x_to_col(x0),y_to_row(y0); c1,r1=x_to_col(x1),y_to_row(y1)
                age=i/(n-1); ch=self.fade_chars[min(int(age*(len(self.fade_chars)-1)),len(self.fade_chars)-1)]
                attr=0
                if self.color_on and self.ramp.enabled: attr=self.ramp.attr_for_fraction(age)
                line(c0,r0,c1,r1,attr,ch)
        try: self.stdscr.addstr(h-1,0,f"XY  t={self._t:7.3f}s  trail={n}  autoscale={'ON' if self.autoscale else 'OFF'}  grid={'ON' if self.grid else 'OFF'}  fps={self.fps}"[:max(0,w-1)])
        except curses.error: pass
        self.stdscr.refresh()
    def run(self):
        self.stdscr.nodelay(True)
        try: curses.curs_set(0)
        except curses.error: pass
        self._resize(); next_frame=time.perf_counter()
        while True:
            try: key=self.stdscr.getch()
            except curses.error: key=-1
            if key!=-1:
                if key in (ord('q'),ord('Q')): break
                elif key==ord(' '): self.paused=not self.paused
                elif key in (ord('a'),ord('A')): self.autoscale=not self.autoscale
                elif key in (ord('g'),ord('G')): self.grid=not self.grid
                elif key in (ord('c'),ord('C')): self.color_on=not self.color_on
                elif key in (ord('r'),ord('R')):
                    if hasattr(self.source,'reset'): self.source.reset()
                    self.trail.clear(); self._resize(); self._t=0.0
                elif key in (ord('1'),ord('2'),ord('3')) and hasattr(self.source,'pair'):
                    self.source.pair={ord('1'):'xy',ord('2'):'xz',ord('3'):'yz'}[key]
            now=time.perf_counter()
            if now<next_frame: time.sleep(max(0,next_frame-now)); continue
            dt_wall=now-getattr(self,'_tick',now); self._tick=now; next_frame=now+1.0/self.fps
            self._resize()
            if not self.paused:
                t_now,pt=self.source.step(dt_wall); self._t=t_now
                x,y=pt
                if math.isfinite(x) and math.isfinite(y): self.trail.append((x,y))
            self._draw()

# ------------------------------- main ---------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Rolling ASCII oscilloscope: 1D & X–Y + MSD + Lorenz + Double Pendulum + Euler spiral")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument('--expr', help="Expression in t for 1D mode, e.g. 'sin(2*pi*2*t)'")
    mode.add_argument('--msd', action='store_true', help='Mass–spring–damper simulator (1D or X–Y with --msd-phase)')
    mode.add_argument('--lorenz', action='store_true', help='Lorenz attractor simulator (1D or X–Y with --xy)')
    mode.add_argument('--pendulum', action='store_true', help='Double pendulum simulator (1D or X–Y)')
    mode.add_argument('--euler', action='store_true', help='Euler (Cornu) spiral (XY only)')

    # XY toggles and specifics
    p.add_argument('--xy', action='store_true', help='Enable X–Y (2D) plot instead of rolling time trace')
    p.add_argument('--xexpr', help='X(t) expression for X–Y expression mode')
    p.add_argument('--yexpr', help='Y(t) expression for X–Y expression mode')
    p.add_argument('--msd-phase', action='store_true', help='In --msd + --xy, plot (x, v) phase portrait')
    p.add_argument('--xy-components', choices=['xy','xz','yz'], default='xy', help='For --lorenz + --xy, choose component pair')

    # MSD params
    p.add_argument('--m', type=float, default=1.0)
    p.add_argument('--c', type=float, default=0.2)
    p.add_argument('--k', type=float, default=10.0)
    p.add_argument('--x0', type=float, default=0.0)
    p.add_argument('--v0', type=float, default=0.0)
    p.add_argument('--force-expr', default="0")

    # Lorenz params
    p.add_argument('--sigma', type=float, default=10.0)
    p.add_argument('--rho', type=float, default=28.0)
    p.add_argument('--beta', type=float, default=8.0/3.0)
    p.add_argument('--x0_l', type=float, default=1.0)
    p.add_argument('--y0_l', type=float, default=1.0)
    p.add_argument('--z0_l', type=float, default=1.0)
    p.add_argument('--component', choices=['x','y','z','t1','t2'], default='x',
                   help='Lorenz: x/y/z in 1D mode; Pendulum: t1/t2 in 1D mode')

    # Double pendulum params
    p.add_argument('--L1', type=float, default=1.0)
    p.add_argument('--L2', type=float, default=1.0)
    p.add_argument('--m1', type=float, default=1.0)
    p.add_argument('--m2', type=float, default=1.0)
    p.add_argument('--g',  type=float, default=9.81)
    p.add_argument('--t1', type=float, default=math.pi/2)
    p.add_argument('--t2', type=float, default=math.pi/2)
    p.add_argument('--w1', type=float, default=0.0)
    p.add_argument('--w2', type=float, default=0.0)

    # Euler spiral params
    p.add_argument('--euler-speed', type=float, default=0.8, help='Parameter speed ds/dt for Euler spiral')

    # display
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--no-autoscale', action='store_true')
    p.add_argument('--no-grid', action='store_true')
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--trail-factor', type=float, default=2.0, help='X–Y trail length vs terminal width (default 2.0)')

    args = p.parse_args()

    # Build source & choose scope
    source=None; scope_class=None; scope_kwargs={}

    if args.xy and not (args.msd or args.lorenz or args.pendulum or args.euler) and (args.xexpr and args.yexpr):
        source=ExprSourceXY(SafeExpr(args.xexpr), SafeExpr(args.yexpr)); scope_class=ScopeXY
        scope_kwargs={'trail_factor': args.trail_factor}

    elif args.expr and not args.xy:
        source=ExprSource1D(SafeExpr(args.expr)); scope_class=Scope1D

    elif args.msd and not args.xy:
        msd=MSDSource1D(m=args.m,c=args.c,k=args.k,x0=args.x0,v0=args.v0,force_expr=SafeExpr(args.force_expr))
        source=msd; scope_class=Scope1D

    elif args.msd and args.xy:
        msd=MSDSource1D(m=args.m,c=args.c,k=args.k,x0=args.x0,v0=args.v0,force_expr=SafeExpr(args.force_expr))
        if not args.msd_phase: p.error("For --msd + --xy you must pass --msd-phase to plot (x,v)")
        source=MSDSourceXY(msd); scope_class=ScopeXY; scope_kwargs={'trail_factor': args.trail_factor}

    elif args.lorenz and not args.xy:
        source=LorenzSource1D(sigma=args.sigma,rho=args.rho,beta=args.beta,
                              x0=args.x0_l,y0=args.y0_l,z0=args.z0_l,component=args.component)
        scope_class=Scope1D

    elif args.lorenz and args.xy:
        base=LorenzSource1D(sigma=args.sigma,rho=args.rho,beta=args.beta,
                            x0=args.x0_l,y0=args.y0_l,z0=args.z0_l,component='x')
        source=LorenzSourceXY(base, pair=args.xy_components); scope_class=ScopeXY
        scope_kwargs={'trail_factor': args.trail_factor}

    elif args.pendulum and not args.xy:
        pend=DoublePendulum1D(L1=args.L1,L2=args.L2,m1=args.m1,m2=args.m2,g=args.g,
                              t1=args.t1,t2=args.t2,w1=args.w1,w2=args.w2,
                              component=args.component if args.component in ('t1','t2') else 't1')
        source=pend; scope_class=Scope1D

    elif args.pendulum and args.xy:
        base=DoublePendulum1D(L1=args.L1,L2=args.L2,m1=args.m1,m2=args.m2,g=args.g,
                              t1=args.t1,t2=args.t2,w1=args.w1,w2=args.w2,component='t1')
        source=DoublePendulumXY(base, which='tip'); scope_class=ScopeXY
        scope_kwargs={'trail_factor': args.trail_factor}

    elif args.euler and args.xy:
        source=EulerSpiralXY(speed=args.euler_speed); scope_class=ScopeXY
        scope_kwargs={'trail_factor': args.trail_factor}

    else:
        p.error("Choose a mode: --expr (1D) or --xy with --xexpr/--yexpr, or --msd, or --lorenz, or --pendulum, or --euler")

    def _run(stdscr):
        if scope_class is Scope1D:
            scope=Scope1D(stdscr=stdscr, source=source, fps=args.fps,
                          autoscale=not args.no_autoscale, grid=not args.no_grid, color_on=not args.no_color)
        else:
            scope=ScopeXY(stdscr=stdscr, source=source, fps=args.fps,
                          autoscale=not args.no_autoscale, grid=not args.no_grid, color_on=not args.no_color,
                          trail_factor=args.trail_factor)
        scope.run()

    try: curses.wrapper(_run)
    except KeyboardInterrupt: pass

if __name__ == '__main__':
    main()
