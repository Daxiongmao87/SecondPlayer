from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...config import SessionConfig
from ...controller import ControllerState, SNES_BUTTONS
from ...errors import AdapterError
from ..base import EmulatorAdapter

log = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1
_ALLOWED_ROM_EXTENSIONS = {".sfc", ".smc"}
_STATE_MAX_BYTES = 64 * 1024 * 1024
_BUTTON_BITS = {button: bit for bit, button in enumerate(SNES_BUTTONS)}


@dataclass(slots=True)
class MesenCEOptions:
    binary: str = "Mesen"
    headless: bool = False
    xvfb: bool = True
    startup_timeout_s: float = 20.0
    capture_timeout_s: float = 5.0
    testrunner_timeout_s: int = 3600
    keep_session: bool = False
    source_config_home: str | None = None
    session_root: str | None = None
    deterministic: bool = False
    pulse_frames: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "MesenCEOptions":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise AdapterError(f"unknown MesenCE adapter option(s): {', '.join(unknown)}")
        values = dict(raw)
        for name in ("headless", "xvfb", "keep_session", "deterministic"):
            if name in values and not isinstance(values[name], bool):
                raise AdapterError(f"adapter.options.{name} must be true or false")
        for name in ("startup_timeout_s", "capture_timeout_s"):
            if name in values:
                values[name] = float(values[name])
                if values[name] <= 0:
                    raise AdapterError(f"adapter.options.{name} must be > 0")
        if "testrunner_timeout_s" in values:
            values["testrunner_timeout_s"] = int(values["testrunner_timeout_s"])
            if values["testrunner_timeout_s"] <= 0:
                raise AdapterError("adapter.options.testrunner_timeout_s must be > 0")
        if "pulse_frames" in values:
            values["pulse_frames"] = int(values["pulse_frames"])
            if values["pulse_frames"] < 0:
                raise AdapterError("adapter.options.pulse_frames must be >= 0")
        return cls(**values)


class MesenCEAdapter(EmulatorAdapter):
    """Stock-MesenCE adapter using its documented Lua scripting interfaces.

    MesenCE remains an external process. The bundled Lua bridge is copied into a
    per-session directory and communicates with this adapter over localhost TCP.
    The user's normal MesenCE configuration is never modified.
    """

    def __init__(self, options: MesenCEOptions, session: SessionConfig, rom: str | Path):
        self.options = options
        self.session = session
        self.rom = Path(rom).expanduser().resolve()
        self._root: Path | None = None
        self._home: Path | None = None
        self._mesen_config: Path | None = None
        self._bridge_path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._listener: socket.socket | None = None
        self._sock: socket.socket | None = None
        self._reader = None
        self._request_id = 0
        self._input_mode: str | None = None
        self._validate_static()

    @property
    def game_name(self) -> str:
        # Human-readable title bait for the model's world knowledge; purely
        # mechanical normalization, the same for every game.
        return self.rom.stem.replace("_", " ").replace("-", " ").strip()

    @property
    def session_root(self) -> Path | None:
        return self._root

    @classmethod
    def doctor(cls, options: MesenCEOptions, session: SessionConfig) -> list[tuple[str, bool, str]]:
        out: list[tuple[str, bool, str]] = []
        binary = cls._resolve_binary(options.binary)
        out.append(("MesenCE", binary is not None, str(binary) if binary else f"not found: {options.binary}"))

        unsupported = sorted(p for p in session.active_players if p not in (1, 2))
        out.append((
            "MesenCE seats",
            not unsupported,
            "SNES ports 1-2" if not unsupported else f"v1 supports only players 1-2, got {unsupported}",
        ))

        if options.headless and options.xvfb:
            xvfb = shutil.which("xvfb-run")
            out.append(("xvfb-run", xvfb is not None, xvfb or "not found"))

        source = cls._source_config_home(options)
        out.append((
            "MesenCE config",
            True,
            f"will copy {source / 'settings.json'}" if (source / "settings.json").is_file() else "no existing config; using isolated minimal settings",
        ))
        return out

    def _validate_static(self) -> None:
        if not self.rom.is_file():
            raise AdapterError(f"ROM not found: {self.rom}")
        if self.rom.suffix.lower() not in _ALLOWED_ROM_EXTENSIONS:
            raise AdapterError("MesenCE v1 adapter currently accepts SNES .sfc/.smc ROMs only")
        unsupported = sorted(p for p in self.session.active_players if p not in (1, 2))
        if unsupported:
            raise AdapterError(f"MesenCE v1 supports only SNES players 1 and 2; unsupported players: {unsupported}")

    @staticmethod
    def _resolve_binary(binary: str) -> Path | None:
        candidate = Path(binary).expanduser()
        if candidate.parent != Path(".") or os.sep in binary:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
            return None
        found = shutil.which(binary)
        return Path(found).resolve() if found else None

    @staticmethod
    def _source_config_home(options: MesenCEOptions) -> Path:
        if options.source_config_home:
            return Path(options.source_config_home).expanduser().resolve()
        return Path.home() / ".config" / "MesenCE"

    @staticmethod
    def _deep_merge(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                MesenCEAdapter._deep_merge(dst[key], value)
            else:
                dst[key] = value
        return dst

    def _prepare_session(self) -> None:
        if self.options.session_root:
            root = Path(self.options.session_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._root = root
        else:
            self._root = Path(tempfile.mkdtemp(prefix="secondplayer-mesence-"))
        self._home = self._root / "home"
        self._mesen_config = self._home / ".config" / "MesenCE"
        self._mesen_config.mkdir(parents=True, exist_ok=True)

        source_settings = self._source_config_home(self.options) / "settings.json"
        settings: dict[str, Any] = {}
        if source_settings.is_file():
            try:
                loaded = json.loads(source_settings.read_text(encoding="utf-8"))
            except Exception as exc:
                raise AdapterError(f"could not read MesenCE settings {source_settings}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise AdapterError(f"MesenCE settings root must be an object: {source_settings}")
            settings = loaded

        patch: dict[str, Any] = {
            "Debug": {
                "ScriptWindow": {
                    "AllowIoOsAccess": True,
                    "AllowNetworkAccess": True,
                    "AutoStartScriptOnLoad": True,
                    "ScriptTimeout": 60,
                }
            },
            "Snes": {
                "Port1": {"Type": "SnesController"},
                "Port2": {"Type": "SnesController"},
            },
        }
        if self.options.deterministic:
            patch["Snes"].update({"RamPowerOnState": 1, "EnableRandomPowerOnState": False})
        self._deep_merge(settings, patch)
        (self._mesen_config / "settings.json").write_text(
            json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        bridge_bytes = resources.files(__package__).joinpath("bridge.lua").read_bytes()
        self._bridge_path = self._root / "secondplayer.lua"
        self._bridge_path.write_bytes(bridge_bytes)

    def _bridge_env(self, port: int) -> dict[str, str]:
        assert self._home is not None
        env = os.environ.copy()
        env["HOME"] = str(self._home)
        env["SECONDPLAYER_BRIDGE_HOST"] = "127.0.0.1"
        env["SECONDPLAYER_BRIDGE_PORT"] = str(port)
        env["SECONDPLAYER_PULSE_FRAMES"] = str(self.options.pulse_frames)
        env["DOTNET_ROLL_FORWARD"] = env.get("DOTNET_ROLL_FORWARD", "Major")
        return env

    def _build_command(self, binary: Path) -> list[str]:
        assert self._bridge_path is not None
        if self.options.headless:
            cmd = [
                str(binary),
                "--testrunner",
                str(self.rom),
                str(self._bridge_path),
                f"--timeout={self.options.testrunner_timeout_s}",
            ]
        else:
            cmd = [str(binary), "--doNotSaveSettings", str(self.rom), str(self._bridge_path)]
        if self.options.xvfb and not os.environ.get("DISPLAY"):
            xvfb = shutil.which("xvfb-run")
            if not xvfb:
                raise AdapterError("no X display and xvfb-run is not installed")
            cmd = [xvfb, "-a", *cmd]
        return cmd

    def start(self) -> None:
        if self._process is not None:
            raise AdapterError("MesenCE adapter already started")
        binary = self._resolve_binary(self.options.binary)
        if binary is None:
            raise AdapterError(f"MesenCE binary not found or not executable: {self.options.binary}")
        self._prepare_session()
        assert self._root is not None and self._home is not None

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(0.2)
        self._listener = listener
        port = int(listener.getsockname()[1])

        env = self._bridge_env(port)

        stdout = (self._root / "mesen.stdout.log").open("wb")
        stderr = (self._root / "mesen.stderr.log").open("wb")
        cmd = self._build_command(binary)
        log.info("launching stock MesenCE: %s", cmd)
        # Any failure from here on must not leak the child process, the bridge
        # listener, or the session directory.
        try:
            try:
                self._process = subprocess.Popen(
                    cmd,
                    cwd=str(self._root),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    # Own process group: under xvfb-run the emulator is a
                    # grandchild, so close() must signal the group, not just
                    # the wrapper, or MesenCE leaks on every run.
                    start_new_session=True,
                )
            finally:
                stdout.close()
                stderr.close()

            deadline = time.monotonic() + self.options.startup_timeout_s
            conn: socket.socket | None = None
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise self._startup_error(f"MesenCE exited before the Lua bridge connected (code {self._process.returncode})")
                try:
                    conn, _ = listener.accept()
                    break
                except socket.timeout:
                    continue
            if conn is None:
                raise self._startup_error("timed out waiting for the MesenCE Lua bridge")

            conn.settimeout(self.options.capture_timeout_s)
            self._sock = conn
            self._reader = conn.makefile("rb")
            hello = self._readline()
            parts = hello.split()
            if len(parts) != 2 or parts[0] != "HELLO" or parts[1] != str(_PROTOCOL_VERSION):
                raise self._startup_error(f"unexpected bridge greeting: {hello!r}")
        except BaseException:
            self.close()
            raise
        log.info("MesenCE Lua bridge connected")

    def _startup_error(self, message: str) -> AdapterError:
        detail = message
        if self._root is not None:
            err = self._root / "mesen.stderr.log"
            if err.is_file():
                tail = err.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
                if tail:
                    detail += f"\nMesenCE stderr:\n{tail}"
        return AdapterError(detail)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._sock is not None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _sendline(self, line: str) -> None:
        if self._sock is None:
            raise AdapterError("MesenCE bridge is not connected")
        try:
            self._sock.sendall(line.encode("ascii") + b"\n")
        except OSError as exc:
            raise AdapterError(f"MesenCE bridge send failed: {exc}") from exc

    def _readline(self) -> str:
        if self._reader is None:
            raise AdapterError("MesenCE bridge is not connected")
        try:
            raw = self._reader.readline()
        except OSError as exc:
            raise AdapterError(f"MesenCE bridge receive failed: {exc}") from exc
        if not raw:
            raise AdapterError("MesenCE bridge closed the connection")
        try:
            return raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise AdapterError("MesenCE bridge returned a non-text protocol header") from exc

    def _read_exact(self, length: int) -> bytes:
        if self._reader is None:
            raise AdapterError("MesenCE bridge is not connected")
        data = self._reader.read(length)
        if data is None or len(data) != length:
            raise AdapterError(f"short frame payload from MesenCE bridge: expected {length}, got {len(data or b'')}")
        return data

    def _response_line(self, expected_id: int, expected_kind: str) -> list[str]:
        while True:
            line = self._readline()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "MODE" and len(parts) >= 2:
                self._input_mode = parts[1]
                log.info("MesenCE setInput compatibility mode: %s", self._input_mode)
                continue
            if parts[0] == "ERROR":
                raise AdapterError("MesenCE Lua bridge error: " + line[6:])
            if len(parts) >= 2 and parts[0] == expected_kind and parts[1] == str(expected_id):
                return parts
            log.debug("ignoring unsolicited MesenCE bridge line: %s", line)

    def capture(self) -> np.ndarray:
        request_id = self._next_id()
        self._sendline(f"CAPTURE {request_id}")
        parts = self._response_line(request_id, "FRAME")
        if len(parts) != 3:
            raise AdapterError(f"invalid FRAME header: {' '.join(parts)}")
        try:
            length = int(parts[2])
        except ValueError as exc:
            raise AdapterError(f"invalid FRAME payload length: {parts[2]!r}") from exc
        if length <= 8 or length > 16 * 1024 * 1024:
            raise AdapterError(f"unreasonable MesenCE screenshot payload size: {length}")
        png = self._read_exact(length)
        try:
            with Image.open(BytesIO(png)) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except Exception as exc:
            raise AdapterError(f"could not decode MesenCE screenshot: {exc}") from exc
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise AdapterError(f"MesenCE returned an invalid RGB frame shape: {frame.shape}")
        return frame

    def wait_frames(self, frames: int) -> None:
        """Block until exactly ``frames`` emulated frames elapse.

        Unlike repeated capture() calls, this advances a precise frame count
        regardless of daemon-side latency, which keeps scripted input prefix
        frame-exact and benchmark starts deterministic.
        """
        if frames < 1:
            raise AdapterError("wait_frames requires at least 1 frame")
        request_id = self._next_id()
        self._sendline(f"WAITFRAMES {request_id} {frames}")
        self._response_line(request_id, "WAITED")

    def save_state(self, path: str | Path) -> bytes:
        """Capture a full-system savestate, write it to ``path``, return bytes.

        The blob is transported length-prefixed and is binary-safe: it may
        contain any byte value, including newlines and NULs.
        """
        request_id = self._next_id()
        self._sendline(f"SAVESTATE {request_id}")
        parts = self._response_line(request_id, "STATE")
        if len(parts) != 3:
            raise AdapterError(f"invalid STATE header: {' '.join(parts)}")
        try:
            length = int(parts[2])
        except ValueError as exc:
            raise AdapterError(f"invalid STATE payload length: {parts[2]!r}") from exc
        if length <= 0 or length > _STATE_MAX_BYTES:
            raise AdapterError(f"unreasonable MesenCE savestate payload size: {length}")
        blob = self._read_exact(length)
        Path(path).write_bytes(blob)
        return blob

    def load_state(self, state: bytes) -> None:
        """Restore a savestate previously returned by :meth:`save_state`.

        The blob is uploaded raw after a length header; no pipelining is used,
        so embedded newlines and NULs pass through untouched.
        """
        if not state:
            raise AdapterError("cannot load an empty MesenCE savestate")
        if len(state) > _STATE_MAX_BYTES:
            raise AdapterError(f"MesenCE savestate too large: {len(state)} bytes")
        if self._sock is None:
            raise AdapterError("MesenCE bridge is not connected")
        request_id = self._next_id()
        try:
            self._sock.sendall(f"LOADSTATE {request_id} {len(state)}\n".encode("ascii") + state)
        except OSError as exc:
            raise AdapterError(f"MesenCE bridge send failed: {exc}") from exc
        self._response_line(request_id, "OK")

    @staticmethod
    def _mask(state: ControllerState) -> int:
        mask = 0
        for button in state.buttons:
            mask |= 1 << _BUTTON_BITS[button]
        return mask

    def apply(self, player: int, state: ControllerState) -> None:
        if player not in (1, 2):
            raise AdapterError(f"MesenCE v1 can apply only SNES players 1 and 2, got {player}")
        self._sendline(f"INPUT {player - 1} {self._mask(state)}")

    def _get_input_mask(self, player: int) -> int:
        request_id = self._next_id()
        self._sendline(f"GETINPUT {request_id} {player - 1}")
        parts = self._response_line(request_id, "INPUTSTATE")
        if len(parts) != 3:
            raise AdapterError(f"invalid INPUTSTATE response: {' '.join(parts)}")
        return int(parts[2])

    @staticmethod
    def _signal_group(pid: int, sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def close(self) -> None:
        process = self._process
        if self._sock is not None:
            try:
                request_id = self._next_id()
                self._sendline(f"SHUTDOWN {request_id}")
            except Exception:
                pass
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._signal_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._signal_group(process.pid, signal.SIGKILL)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass

        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
        self._reader = None
        self._sock = None
        self._listener = None
        self._process = None

        if self._root is not None and not self.options.keep_session and not self.options.session_root:
            shutil.rmtree(self._root, ignore_errors=True)
