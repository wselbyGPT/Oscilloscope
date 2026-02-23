#!/usr/bin/env python3
"""
pyssl - a tiny "openssl-like" CLI toolkit (educational).

Commands:
  - rand   : random bytes (hex/base64/raw)
  - dgst   : hashes + optional HMAC
  - enc    : AES-256-GCM encrypt/decrypt w/ password via scrypt
  - genpkey: RSA or EC key generation (PEM)
  - pkey   : key info + SHA256 fingerprint
  - sign   : RSA-PSS or ECDSA signatures
  - verify : verify signatures
  - x509   : inspect certs
  - scope  : live curses oscilloscope of bytes with ANSI gradient

Note:
  - dgst/rand work with stdlib only.
  - enc/genpkey/sign/verify/x509 require `cryptography`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import curses
import getpass
import hashlib
import hmac
import math
import os
import secrets
import select
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

MAGIC = b"PYSSL1"  # header for pyssl enc blobs
SALT_LEN = 16
NONCE_LEN = 12

# --- Optional dependency (needed for enc/genpkey/sign/verify/x509) ---
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
except Exception:  # pragma: no cover
    x509 = None
    hashes = None
    serialization = None
    ec = None
    padding = None
    rsa = None
    AESGCM = None
    Scrypt = None


def die(msg: str, code: int = 2) -> None:
    print(f"pyssl: {msg}", file=sys.stderr)
    raise SystemExit(code)


def read_bytes(path: Optional[str]) -> bytes:
    if path is None or path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def write_bytes(path: Optional[str], data: bytes) -> None:
    if path is None or path == "-":
        sys.stdout.buffer.write(data)
        return
    Path(path).write_bytes(data)


def write_text(path: Optional[str], text: str) -> None:
    if path is None or path == "-":
        sys.stdout.write(text)
        return
    Path(path).write_text(text, encoding="utf-8")


def hexdump(data: bytes) -> str:
    return binascii.hexlify(data).decode("ascii")


# -------------------------
# rand
# -------------------------
def cmd_rand(args: argparse.Namespace) -> None:
    data = secrets.token_bytes(args.num_bytes)
    if args.hex:
        out = hexdump(data) + ("" if args.no_newline else "\n")
        write_text(args.out, out)
    elif args.base64:
        out = base64.b64encode(data).decode("ascii") + ("" if args.no_newline else "\n")
        write_text(args.out, out)
    else:
        write_bytes(args.out, data)


# -------------------------
# dgst (hash / HMAC)
# -------------------------
_HASH_ALGOS = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha224": hashlib.sha224,
    "sha256": hashlib.sha256,
    "sha384": hashlib.sha384,
    "sha512": hashlib.sha512,
    "blake2b": hashlib.blake2b,
    "blake2s": hashlib.blake2s,
}


def _pick_hash_algo(args: argparse.Namespace) -> str:
    for a in ["md5", "sha1", "sha224", "sha256", "sha384", "sha512", "blake2b", "blake2s"]:
        if getattr(args, a, False):
            return a
    return args.algo


def cmd_dgst(args: argparse.Namespace) -> None:
    algo = _pick_hash_algo(args)
    if algo not in _HASH_ALGOS:
        die(f"unsupported digest algorithm: {algo}")

    data = read_bytes(args.infile)

    if args.hmac is not None:
        key = args.hmac.encode("utf-8")
        digest = hmac.new(key, data, _HASH_ALGOS[algo]).digest()
    else:
        digest = _HASH_ALGOS[algo](data).digest()

    if args.raw:
        write_bytes(args.out, digest)
        return

    if args.base64:
        out = base64.b64encode(digest).decode("ascii")
    else:
        out = hexdump(digest)

    if args.openssl_style:
        name = f"{'HMAC-' if args.hmac is not None else ''}{algo.upper()}"
        target = args.infile if args.infile and args.infile != "-" else "stdin"
        out = f"{name}({target})= {out}"

    if not args.no_newline:
        out += "\n"
    write_text(args.out, out)


# -------------------------
# enc (AES-256-GCM + scrypt)
# -------------------------
@dataclass
class EncHeader:
    salt: bytes
    nonce: bytes

    def pack(self) -> bytes:
        return MAGIC + self.salt + self.nonce

    @staticmethod
    def unpack(blob: bytes) -> Tuple["EncHeader", bytes]:
        min_len = len(MAGIC) + SALT_LEN + NONCE_LEN
        if len(blob) < min_len:
            die("ciphertext too short / missing header")
        if blob[: len(MAGIC)] != MAGIC:
            die("bad file header (not a pyssl enc blob)")
        salt = blob[len(MAGIC) : len(MAGIC) + SALT_LEN]
        nonce = blob[len(MAGIC) + SALT_LEN : min_len]
        rest = blob[min_len:]
        return EncHeader(salt=salt, nonce=nonce), rest


def _get_password(pass_spec: Optional[str], confirm: bool) -> bytes:
    if pass_spec is None:
        pw = getpass.getpass("password: ")
        if confirm:
            pw2 = getpass.getpass("confirm: ")
            if pw != pw2:
                die("passwords do not match")
        return pw.encode("utf-8")

    if pass_spec.startswith("pass:"):
        return pass_spec[5:].encode("utf-8")
    if pass_spec.startswith("env:"):
        var = pass_spec[4:]
        val = os.environ.get(var)
        if val is None:
            die(f"environment variable not set: {var}")
        return val.encode("utf-8")
    if pass_spec.startswith("file:"):
        p = pass_spec[5:]
        return Path(p).read_text(encoding="utf-8").rstrip("\n").encode("utf-8")
    die("unsupported -pass format (use pass:..., env:..., or file:...)")


def _derive_key_scrypt(password: bytes, salt: bytes) -> bytes:
    if Scrypt is None:
        die("cryptography not installed (required for enc). Install with: pip install cryptography")
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(password)


def cmd_enc(args: argparse.Namespace) -> None:
    if AESGCM is None:
        die("cryptography not installed (required for enc). Install with: pip install cryptography")

    inp = read_bytes(args.infile)

    if args.base64_in:
        try:
            inp = base64.b64decode(inp, validate=True)
        except Exception:
            die("failed to base64-decode input (-a/--base64-in)")

    encrypting = args.encrypt
    password = _get_password(args.passphrase, confirm=encrypting)

    if encrypting:
        salt = secrets.token_bytes(SALT_LEN)
        nonce = secrets.token_bytes(NONCE_LEN)
        key = _derive_key_scrypt(password, salt)
        ct = AESGCM(key).encrypt(nonce, inp, associated_data=None)
        blob = EncHeader(salt=salt, nonce=nonce).pack() + ct
    else:
        hdr, ct = EncHeader.unpack(inp)
        key = _derive_key_scrypt(password, hdr.salt)
        try:
            blob = AESGCM(key).decrypt(hdr.nonce, ct, associated_data=None)
        except Exception:
            die("decryption failed (wrong password? corrupted data?)")

    if args.base64_out:
        out = base64.b64encode(blob) + (b"" if args.no_newline else b"\n")
        write_bytes(args.out, out)
    else:
        write_bytes(args.out, blob)


# -------------------------
# key generation / loading
# -------------------------
def _need_crypto(feature: str) -> None:
    if serialization is None:
        die(f"cryptography not installed (required for {feature}). Install with: pip install cryptography")


def load_private_key(path: str, password: Optional[bytes]) -> object:
    _need_crypto("keys")
    data = Path(path).read_bytes()
    try:
        return serialization.load_pem_private_key(data, password=password)
    except ValueError:
        return serialization.load_der_private_key(data, password=password)


def load_public_key(path: str) -> object:
    _need_crypto("keys")
    data = Path(path).read_bytes()
    try:
        return serialization.load_pem_public_key(data)
    except ValueError:
        return serialization.load_der_public_key(data)


def save_private_key(key: object, path: str, password: Optional[bytes]) -> None:
    _need_crypto("keys")
    enc = serialization.NoEncryption()
    if password:
        enc = serialization.BestAvailableEncryption(password)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=enc,
    )
    Path(path).write_bytes(pem)


def save_public_key(key: object, path: str) -> None:
    _need_crypto("keys")
    pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    Path(path).write_bytes(pem)


def cmd_genpkey(args: argparse.Namespace) -> None:
    if rsa is None or ec is None:
        die("cryptography not installed (required for genpkey). Install with: pip install cryptography")

    password = None
    if args.encrypt_key:
        password = getpass.getpass("private key password: ").encode("utf-8")

    if args.algorithm == "rsa":
        key = rsa.generate_private_key(public_exponent=65537, key_size=args.bits)
    elif args.algorithm == "ec":
        curve_map = {
            "secp256r1": ec.SECP256R1(),
            "secp384r1": ec.SECP384R1(),
            "secp521r1": ec.SECP521R1(),
        }
        curve = curve_map.get(args.curve)
        if curve is None:
            die(f"unsupported curve: {args.curve} (try: {', '.join(curve_map)})")
        key = ec.generate_private_key(curve)
    else:
        die("unknown algorithm")

    save_private_key(key, args.out, password=password)
    if args.pubout:
        save_public_key(key, args.pubout)


def key_info(key: object) -> str:
    if rsa and isinstance(key, (rsa.RSAPrivateKey, rsa.RSAPublicKey)):
        return f"RSA {key.key_size}-bit"
    if ec and isinstance(key, (ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey)):
        return f"EC {key.curve.name}"
    return type(key).__name__


def cmd_pkey(args: argparse.Namespace) -> None:
    _need_crypto("pkey")

    data = Path(args.infile).read_bytes()
    obj = None
    is_priv = False
    try:
        obj = serialization.load_pem_private_key(data, password=None)
        is_priv = True
    except Exception:
        try:
            obj = serialization.load_pem_public_key(data)
            is_priv = False
        except Exception:
            die("could not parse key (PEM private/public). If it's encrypted, use sign with -keypass.")

    pub = obj.public_key() if is_priv else obj
    der = pub.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    fp = hashlib.sha256(der).hexdigest()
    fp_colon = ":".join(fp[i : i + 2] for i in range(0, len(fp), 2))

    out = (
        f"Type: {'private' if is_priv else 'public'}\n"
        f"Algo: {key_info(obj)}\n"
        f"Public key SHA256 fingerprint: {fp_colon}\n"
    )
    write_text(args.out, out)


# -------------------------
# sign / verify
# -------------------------
def parse_pass_spec(spec: Optional[str]) -> Optional[bytes]:
    if spec is None:
        return None
    if spec.startswith("pass:"):
        return spec[5:].encode("utf-8")
    if spec.startswith("env:"):
        var = spec[4:]
        val = os.environ.get(var)
        if val is None:
            die(f"environment variable not set: {var}")
        return val.encode("utf-8")
    if spec.startswith("file:"):
        return Path(spec[5:]).read_text(encoding="utf-8").rstrip("\n").encode("utf-8")
    die("unsupported password format (use pass:..., env:..., file:...)")


def cmd_sign(args: argparse.Namespace) -> None:
    if padding is None or hashes is None:
        die("cryptography not installed (required for sign). Install with: pip install cryptography")

    priv = load_private_key(args.key, password=parse_pass_spec(args.key_pass))
    data = read_bytes(args.infile)

    if rsa and isinstance(priv, rsa.RSAPrivateKey):
        sig = priv.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        alg = "rsa-pss-sha256"
    elif ec and isinstance(priv, ec.EllipticCurvePrivateKey):
        sig = priv.sign(data, ec.ECDSA(hashes.SHA256()))
        alg = "ecdsa-sha256"
    else:
        die("unsupported private key type for signing (RSA or EC)")

    if args.base64:
        out = base64.b64encode(sig).decode("ascii")
        if args.with_meta:
            out = f"{alg}:{out}"
        if not args.no_newline:
            out += "\n"
        write_text(args.out, out)
    else:
        write_bytes(args.out, sig)


def _parse_sig_input(raw: bytes, base64_mode: bool) -> bytes:
    if not base64_mode:
        return raw
    s = raw.decode("utf-8").strip()
    # Accept optional "alg:" prefix
    if ":" in s and s.split(":", 1)[0].startswith(("rsa", "ecdsa")):
        s = s.split(":", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        die("failed to base64-decode signature")


def cmd_verify(args: argparse.Namespace) -> None:
    if padding is None or hashes is None:
        die("cryptography not installed (required for verify). Install with: pip install cryptography")

    pub = load_public_key(args.pubkey)
    data = read_bytes(args.infile)
    sig_raw = read_bytes(args.signature)
    sig = _parse_sig_input(sig_raw, args.base64)

    ok = False
    try:
        if rsa and isinstance(pub, rsa.RSAPublicKey):
            pub.verify(
                sig,
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            ok = True
        elif ec and isinstance(pub, ec.EllipticCurvePublicKey):
            pub.verify(sig, data, ec.ECDSA(hashes.SHA256()))
            ok = True
        else:
            die("unsupported public key type for verify (RSA or EC)")
    except Exception:
        ok = False

    if args.quiet:
        raise SystemExit(0 if ok else 1)
    print("Verified OK" if ok else "Verification FAILED")
    raise SystemExit(0 if ok else 1)


# -------------------------
# x509 inspect
# -------------------------
def cmd_x509(args: argparse.Namespace) -> None:
    if x509 is None or serialization is None:
        die("cryptography not installed (required for x509). Install with: pip install cryptography")

    data = Path(args.infile).read_bytes()
    try:
        cert = x509.load_pem_x509_certificate(data)
    except Exception:
        cert = x509.load_der_x509_certificate(data)

    pub = cert.public_key()
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    fp = hashlib.sha256(cert_der).hexdigest()
    fp_colon = ":".join(fp[i : i + 2] for i in range(0, len(fp), 2))

    nb = getattr(cert, "not_valid_before_utc", cert.not_valid_before)
    na = getattr(cert, "not_valid_after_utc", cert.not_valid_after)

    out_lines = [
        f"Subject: {cert.subject.rfc4514_string()}",
        f"Issuer:  {cert.issuer.rfc4514_string()}",
        f"Serial:  {cert.serial_number}",
        f"Not Before: {nb.isoformat()}",
        f"Not After:  {na.isoformat()}",
        f"Public Key: {key_info(pub)}",
        f"Cert SHA256 fingerprint: {fp_colon}",
    ]

    if args.pubout:
        pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        Path(args.pubout).write_bytes(pem)

    write_text(args.out, "\n".join(out_lines) + "\n")


# -------------------------
# scope (curses oscilloscope)
# -------------------------
ASCII_RAMP = " .:-=+*#%@"
ASCII_RAMP_LEN = len(ASCII_RAMP)


def _init_color_ramp(palette: str) -> list[int]:
    """Return a list of color-pair IDs from dim->bright."""
    curses.start_color()
    try:
        curses.use_default_colors()
    except Exception:
        pass

    pairs: list[int] = []
    max_pairs = max(0, curses.COLOR_PAIRS - 1)

    if curses.COLORS >= 256:
        ramps = {
            "neon": [17, 18, 19, 20, 21, 27, 33, 39, 45, 51, 50, 49, 48, 47, 46, 82, 118, 154, 190, 226, 220, 214, 208, 202, 196],
            "fire": [52, 88, 124, 160, 196, 202, 208, 214, 220, 226],
            "ocean": [17, 18, 19, 20, 21, 27, 33, 39, 45, 51],
            "mono": [244, 246, 248, 250, 252, 254, 255],
        }
        ramp = ramps.get(palette, ramps["neon"])
        for i, c in enumerate(ramp, start=1):
            if i > max_pairs:
                break
            curses.init_pair(i, c, -1)
            pairs.append(i)
    else:
        ramps = {
            "neon": [curses.COLOR_BLUE, curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_MAGENTA],
            "fire": [curses.COLOR_RED, curses.COLOR_YELLOW, curses.COLOR_WHITE],
            "ocean": [curses.COLOR_BLUE, curses.COLOR_CYAN, curses.COLOR_WHITE],
            "mono": [curses.COLOR_WHITE],
        }
        ramp = ramps.get(palette, ramps["neon"])
        for i, c in enumerate(ramp, start=1):
            if i > max_pairs:
                break
            curses.init_pair(i, c, -1)
            pairs.append(i)

    if not pairs:
        pairs = [0]
    return pairs


class _ByteSource:
    def __init__(self, mode: str, infile: str | None, loop: bool, chunk: int):
        self.mode = mode  # "rand" | "file" | "stdin"
        self.loop = loop
        self.chunk = chunk
        self._fp = None
        self._fd = None

        if mode == "rand":
            return

        if infile is None:
            infile = "-"

        if mode == "stdin":
            infile = "-"

        if infile == "-":
            self._fp = sys.stdin.buffer
            self._fd = self._fp.fileno()
        else:
            self._fp = open(infile, "rb", buffering=0)
            self._fd = self._fp.fileno()

    def close(self) -> None:
        try:
            if self._fp not in (None, sys.stdin.buffer):
                self._fp.close()
        except Exception:
            pass

    def read_some(self) -> bytes:
        if self.mode == "rand":
            return secrets.token_bytes(self.chunk)

        if self._fp is None or self._fd is None:
            return b""

        # For pipes, avoid blocking.
        try:
            r, _, _ = select.select([self._fd], [], [], 0.0)
            if not r:
                return b""
        except Exception:
            # If select fails, fallback to a small blocking read.
            try:
                return self._fp.read(self.chunk) or b""
            except Exception:
                return b""

        try:
            b = os.read(self._fd, self.chunk)
        except Exception:
            try:
                b = self._fp.read(self.chunk) or b""
            except Exception:
                b = b""

        # If file EOF and looping enabled, rewind.
        if b == b"" and self.loop and self._fp not in (None, sys.stdin.buffer):
            try:
                self._fp.seek(0)
                b = self._fp.read(self.chunk) or b""
            except Exception:
                return b""
        return b


def _entropy_bits(window: list[int]) -> float:
    if not window:
        return 0.0
    counts = [0] * 256
    for v in window:
        counts[v] += 1
    n = len(window)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def _safe_add(stdscr, y: int, x: int, s: str, attr: int = 0) -> None:
    maxy, maxx = stdscr.getmaxyx()
    if 0 <= y < maxy and 0 <= x < maxx:
        try:
            stdscr.addstr(y, x, s[: maxx - x], attr)
        except Exception:
            pass


def _run_curses_app(app_fn, args) -> None:
    """
    Supports piped stdin by taking keystrokes from /dev/tty.
    Data can still come from stdin pipe via sys.stdin.buffer.
    """
    if not sys.stdout.isatty():
        die("scope requires a TTY stdout (don’t redirect output)")

    tty_in = None

    # If stdin isn’t a TTY (piped), open /dev/tty for key controls.
    if not sys.stdin.isatty():
        try:
            tty_in = open("/dev/tty", "rb", buffering=0)
        except Exception:
            tty_in = None

    if tty_in is not None:
        term = os.environ.get("TERM") or "xterm-256color"
        stdscr = curses.newterm(term, sys.stdout, tty_in)  # returns stdscr window
        curses.set_term(stdscr)  # ensure it's current
        try:
            curses.noecho()
            curses.cbreak()
            try:
                curses.curs_set(0)
            except Exception:
                pass
            stdscr.keypad(True)
            app_fn(stdscr, args)
        finally:
            try:
                stdscr.keypad(False)
            except Exception:
                pass
            try:
                curses.nocbreak()
                curses.echo()
            except Exception:
                pass
            curses.endwin()
            try:
                tty_in.close()
            except Exception:
                pass
    else:
        curses.wrapper(app_fn, args)


def cmd_scope(args: argparse.Namespace) -> None:
    # Normalize source/infile combos
    if args.source == "stdin":
        args.infile = "-"
    elif args.source == "file":
        if args.infile in (None, "-"):
            die("scope --source file requires -in <path>")
    # rand ignores infile

    _run_curses_app(_scope_main, args)


def _scope_main(stdscr, args: argparse.Namespace) -> None:
    stdscr.nodelay(True)
    stdscr.timeout(0)

    pairs = _init_color_ramp(args.palette)
    pair_n = len(pairs)

    src = _ByteSource(mode=args.source, infile=args.infile, loop=args.loop, chunk=args.chunk)

    max_hist = max(4096, args.history)
    buf = deque([0] * min(512, max_hist), maxlen=max_hist)

    paused = False
    mode = args.mode  # "wave" | "hist" | "both"
    fps = max(5, min(120, args.fps))

    total_bytes = 0
    last_t = time.time()
    last_bytes = 0
    rate_bps = 0.0

    help_line = "q quit | space pause | m mode | p palette | +/- fps | l loop | r reset"
    palettes = ["neon", "fire", "ocean", "mono"]
    pal_idx = palettes.index(args.palette) if args.palette in palettes else 0

    while True:
        now = time.time()
        dt = now - last_t
        if dt >= 0.25:
            rate_bps = (total_bytes - last_bytes) / dt
            last_bytes = total_bytes
            last_t = now

        # keys
        try:
            ch = stdscr.getch()
        except Exception:
            ch = -1

        if ch != -1:
            if ch in (ord("q"), ord("Q")):
                break
            elif ch == ord(" "):
                paused = not paused
            elif ch in (ord("m"), ord("M")):
                mode = {"wave": "hist", "hist": "both", "both": "wave"}[mode]
            elif ch in (ord("p"), ord("P")):
                pal_idx = (pal_idx + 1) % len(palettes)
                pairs = _init_color_ramp(palettes[pal_idx])
                pair_n = len(pairs)
            elif ch in (ord("+"), ord("=")):
                fps = min(120, fps + 5)
            elif ch in (ord("-"), ord("_")):
                fps = max(5, fps - 5)
            elif ch in (ord("l"), ord("L")):
                args.loop = not args.loop
                src.loop = args.loop
            elif ch in (ord("r"), ord("R")):
                buf.clear()
                buf.extend([0] * min(512, max_hist))
                total_bytes = 0

        # data
        if not paused:
            b = src.read_some()
            if b:
                total_bytes += len(b)
                for v in b:
                    buf.append(v)

        # draw
        stdscr.erase()
        maxy, maxx = stdscr.getmaxyx()

        src_label = args.source if args.source == "rand" else f"{args.source}:{args.infile or '-'}"
        title = f"pyssl scope | mode={mode} | src={src_label}"
        stats = f"bytes={total_bytes}  rate={rate_bps:,.0f} B/s  fps={fps}  loop={'on' if args.loop else 'off'}  palette={palettes[pal_idx]}"
        window_list = list(buf)[-min(len(buf), args.entropy_window):]
        ent = _entropy_bits(window_list)

        _safe_add(stdscr, 0, 0, title, curses.A_BOLD)
        _safe_add(stdscr, 1, 0, stats)
        _safe_add(stdscr, 2, 0, f"entropy≈{ent:0.2f} bits/byte (recent {len(window_list)} bytes)   {help_line}")

        top = 4
        bottom = maxy - 1
        if bottom <= top + 2:
            stdscr.refresh()
            time.sleep(0.05)
            continue

        # layout
        if mode == "wave":
            wave_top, wave_bot = top, bottom
            hist_top, hist_bot = 0, -1
        elif mode == "hist":
            wave_top, wave_bot = 0, -1
            hist_top, hist_bot = top, bottom
        else:
            split = top + (bottom - top) * 2 // 3
            wave_top, wave_bot = top, split
            hist_top, hist_bot = split + 1, bottom

        # waveform
        if wave_bot > wave_top:
            h = wave_bot - wave_top + 1
            w = max(2, maxx - 2)

            view = list(buf)[-w * args.xscale :]
            if not view:
                view = [0]

            cols = []
            step = max(1, len(view) // w)
            for i in range(0, len(view), step):
                chunk = view[i : i + step]
                cols.append(sum(chunk) / len(chunk))
                if len(cols) >= w:
                    break
            if len(cols) < w:
                cols += [cols[-1]] * (w - len(cols))

            mid = wave_top + h // 2
            for x in range(1, w + 1):
                _safe_add(stdscr, mid, x, "-", curses.A_DIM)

            for x, v in enumerate(cols, start=1):
                y = wave_bot - int((v / 255.0) * (h - 1))
                inten = v / 255.0
                pair = pairs[min(pair_n - 1, int(inten * (pair_n - 1)))]
                ch_idx = min(ASCII_RAMP_LEN - 1, int(inten * (ASCII_RAMP_LEN - 1)))
                ch_draw = ASCII_RAMP[ch_idx]
                attr = curses.color_pair(pair) | curses.A_BOLD
                _safe_add(stdscr, y, x, ch_draw, attr)

        # histogram (16 bins)
        if hist_bot > hist_top:
            h2 = hist_bot - hist_top + 1
            w2 = max(16, maxx - 2)
            bins = 16
            counts = [0] * bins
            recent = list(buf)[-min(len(buf), args.hist_window):]
            if recent:
                for v in recent:
                    counts[v * bins // 256] += 1
                m = max(counts) or 1
            else:
                m = 1

            bar_w = max(1, w2 // bins)
            for i in range(bins):
                frac = counts[i] / m
                bar_h = int(frac * (h2 - 1))
                inten = i / (bins - 1)
                pair = pairs[min(pair_n - 1, int(inten * (pair_n - 1)))]
                attr = curses.color_pair(pair) | curses.A_BOLD
                x0 = 1 + i * bar_w
                for yy in range(hist_bot, hist_bot - bar_h, -1):
                    for xx in range(x0, min(x0 + bar_w, maxx - 1)):
                        _safe_add(stdscr, yy, xx, "#", attr)

            _safe_add(stdscr, hist_top, 1, f"histogram (last {len(recent)} bytes)", curses.A_DIM)

        stdscr.refresh()
        time.sleep(max(0.0, (1.0 / fps) - 0.001))

    src.close()


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyssl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="A tiny openssl-like toolkit (educational).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # rand
    pr = sub.add_parser("rand", help="Generate random bytes")
    pr.add_argument("num_bytes", type=int, help="Number of bytes")
    fmt = pr.add_mutually_exclusive_group()
    fmt.add_argument("-hex", dest="hex", action="store_true", help="Output hex")
    fmt.add_argument("-base64", dest="base64", action="store_true", help="Output base64")
    pr.add_argument("-out", dest="out", default="-", help="Output file (default stdout)")
    pr.add_argument("-n", "--no-newline", action="store_true", help="No trailing newline (text modes)")
    pr.set_defaults(func=cmd_rand)

    # dgst
    pd = sub.add_parser("dgst", help="Compute digests (hash/HMAC)")
    pd.add_argument("-in", dest="infile", default="-", help="Input file (default stdin)")
    pd.add_argument("-out", dest="out", default="-", help="Output file (default stdout)")
    pd.add_argument("-algo", dest="algo", default="sha256", help="Digest algorithm (default sha256)")
    for a in ["md5", "sha1", "sha224", "sha256", "sha384", "sha512", "blake2b", "blake2s"]:
        pd.add_argument(f"-{a}", dest=a, action="store_true", help=f"Use {a}")
    pd.add_argument("-hmac", dest="hmac", default=None, help="HMAC key (utf-8)")
    pd.add_argument("-r", "--raw", action="store_true", help="Output raw bytes")
    pd.add_argument("-b64", "--base64", action="store_true", help="Output base64")
    pd.add_argument("-o", "--openssl-style", action="store_true", help="Output like OpenSSL dgst")
    pd.add_argument("-n", "--no-newline", action="store_true", help="No trailing newline (text modes)")
    pd.set_defaults(func=cmd_dgst)

    # enc
    pe = sub.add_parser("enc", help="Encrypt/decrypt (AES-256-GCM + scrypt)")
    mode = pe.add_mutually_exclusive_group(required=True)
    mode.add_argument("-e", dest="encrypt", action="store_true", help="Encrypt")
    mode.add_argument("-d", dest="encrypt", action="store_false", help="Decrypt")
    pe.add_argument("-in", dest="infile", default="-", help="Input file (default stdin)")
    pe.add_argument("-out", dest="out", default="-", help="Output file (default stdout)")
    pe.add_argument(
        "-pass",
        dest="passphrase",
        default=None,
        help="Password source: pass:..., env:..., file:... (otherwise prompt)",
    )
    pe.add_argument("-a", "--base64-in", action="store_true", help="Input is base64 text")
    pe.add_argument("-A", "--base64-out", action="store_true", help="Output as base64 text")
    pe.add_argument("-n", "--no-newline", action="store_true", help="No trailing newline for base64-out")
    pe.set_defaults(func=cmd_enc)

    # genpkey
    pg = sub.add_parser("genpkey", help="Generate private/public keys (PEM)")
    pg.add_argument("-algorithm", choices=["rsa", "ec"], required=True, help="Key algorithm")
    pg.add_argument("-out", required=True, help="Private key output (PEM)")
    pg.add_argument("-pubout", default=None, help="Optional public key output (PEM)")
    pg.add_argument("-encrypt", dest="encrypt_key", action="store_true", help="Encrypt private key with password prompt")
    pg.add_argument("-bits", type=int, default=2048, help="RSA bits (default 2048)")
    pg.add_argument("-curve", default="secp256r1", help="EC curve (default secp256r1)")
    pg.set_defaults(func=cmd_genpkey)

    # pkey
    pk = sub.add_parser("pkey", help="Show key info (PEM)")
    pk.add_argument("-in", dest="infile", required=True, help="Key file (PEM)")
    pk.add_argument("-out", dest="out", default="-", help="Output (default stdout)")
    pk.set_defaults(func=cmd_pkey)

    # sign
    ps = sub.add_parser("sign", help="Sign data with private key (RSA-PSS or ECDSA)")
    ps.add_argument("-key", required=True, help="Private key (PEM)")
    ps.add_argument("-keypass", dest="key_pass", default=None, help="Key password source: pass:..., env:..., file:...")
    ps.add_argument("-in", dest="infile", default="-", help="Input file (default stdin)")
    ps.add_argument("-out", dest="out", default="-", help="Signature output (default stdout)")
    ps.add_argument("-b64", dest="base64", action="store_true", help="Output base64 signature")
    ps.add_argument("--with-meta", action="store_true", help="Prefix base64 signature with 'alg:' tag")
    ps.add_argument("-n", "--no-newline", action="store_true", help="No trailing newline (text mode)")
    ps.set_defaults(func=cmd_sign)

    # verify
    pv = sub.add_parser("verify", help="Verify signature with public key")
    pv.add_argument("-pubkey", required=True, help="Public key (PEM)")
    pv.add_argument("-sig", dest="signature", required=True, help="Signature file ('-' for stdin)")
    pv.add_argument("-in", dest="infile", default="-", help="Input file (default stdin)")
    pv.add_argument("-b64", dest="base64", action="store_true", help="Signature is base64 text (optionally 'alg:...')")
    pv.add_argument("-q", "--quiet", action="store_true", help="Quiet (exit code only)")
    pv.set_defaults(func=cmd_verify)

    # x509
    px = sub.add_parser("x509", help="Inspect X.509 certificate (PEM/DER)")
    px.add_argument("-in", dest="infile", required=True, help="Certificate file")
    px.add_argument("-out", dest="out", default="-", help="Output (default stdout)")
    px.add_argument("-pubout", default=None, help="Write public key (PEM) to file")
    px.set_defaults(func=cmd_x509)

    # scope
    psc = sub.add_parser("scope", help="Live curses scope visualization of bytes")
    psc.add_argument("-in", dest="infile", default="-", help="Input file (default stdin; can be piped)")
    psc.add_argument("--source", choices=["rand", "file", "stdin"], default="rand", help="Data source (default rand)")
    psc.add_argument("--mode", choices=["wave", "hist", "both"], default="both", help="Display mode")
    psc.add_argument("--palette", choices=["neon", "fire", "ocean", "mono"], default="neon", help="Color palette")
    psc.add_argument("--fps", type=int, default=30, help="Frames per second (default 30)")
    psc.add_argument("--chunk", type=int, default=4096, help="Bytes to read per tick (default 4096)")
    psc.add_argument("--history", type=int, default=16384, help="Ring buffer size (default 16384)")
    psc.add_argument("--xscale", type=int, default=8, help="Time stretch (higher = smoother, default 8)")
    psc.add_argument("--entropy-window", type=int, default=2048, help="Entropy window (default 2048)")
    psc.add_argument("--hist-window", type=int, default=4096, help="Histogram window (default 4096)")
    psc.add_argument("--loop", action="store_true", help="Loop file input at EOF")
    psc.set_defaults(func=cmd_scope)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
