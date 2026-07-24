"""lyric_page.py — Syntheticho lyric-style sentence page (offline, local).

A parallel side-channel to whisper_tts.py: instead of speaking the
monologue, it pushes each sentence to a local browser page over
Server-Sent Events, rendered like scrolling lyrics (black background,
glowing thin serif type). Meant for a secondary display in place of a
raw numeric readout.

Mirrors whisper_tts.py's shape on purpose: a module-level singleton
(`lyric_page`) exposing .start() / .push(text, is_reassemble=False) /
.stop(), backed by a daemon thread, and never raising into the caller —
any failure prints one warning via _warn_once and degrades silently.
"""

import os
import time
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from pythonosc import dispatcher as _osc_dispatcher
    from pythonosc import osc_server as _osc_server
    _HAND_OSC_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - degrades silently if python-osc missing
    _osc_dispatcher = None
    _osc_server = None
    _HAND_OSC_IMPORT_ERROR = _e

# ======================= 可调参数 =======================
LYRIC_ENABLE = os.environ.get("LYRIC_ENABLE", "1") == "1"   # 总开关，默认开
LYRIC_PORT = int(os.environ.get("LYRIC_PAGE_PORT", "8137"))
LYRIC_HOST = os.environ.get("LYRIC_PAGE_HOST", "127.0.0.1")
LYRIC_SHOW_REASSEMBLE = os.environ.get("LYRIC_SHOW_REASSEMBLE", "0") == "1"  # 是否显示 reunite 阶段的机械读数句

# hand-state side channel: hand_osc.py (or any OSC sender) posts /handon
# (1.0 = hand present, 0.0 = none) here; we relay it to the page over SSE
# as a named "hand" event, kept separate from the monologue stream.
LYRIC_HAND_OSC_ENABLE = os.environ.get("LYRIC_HAND_OSC_ENABLE", "1") == "1"
LYRIC_HAND_OSC_PORT = int(os.environ.get("LYRIC_HAND_OSC_PORT", "10728"))
LYRIC_HAND_OSC_ADDR = os.environ.get("LYRIC_HAND_OSC_ADDR", "/handon")
# ==========================================================

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH = os.path.join(_PROJECT_DIR, "lyric_page.html")

# Static assets the page references (fonts, logo) beyond / and /stream.
# Allowlisted by path prefix/exact-match, not just "anything under the repo",
# so this server can't be turned into a general file server by accident.
_STATIC_ALLOWED_PREFIXES = ("/fonts/",)
_STATIC_ALLOWED_FILES = ("/logo2.png",)
_STATIC_MIME_TYPES = {
    ".woff2": "font/woff2",
    ".png": "image/png",
}

_FALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>lyric</title>
<style>body{background:#000;color:#fff;font-family:Georgia,serif;
font-weight:300;display:flex;align-items:center;justify-content:center;
height:100vh;margin:0;text-align:center;}</style></head>
<body><div id="line">...</div>
<script>
var line = document.getElementById('line');
var es = new EventSource('/stream');
es.onmessage = function(e){ line.textContent = e.data; };
</script>
</body></html>"""

_SUBSCRIBER_QUEUE_MAX = 20
_HEARTBEAT_SECONDS = 15

# Sentinel tuple shape put on a subscriber queue to mark a hand-state frame,
# distinct from the plain strings the monologue path already queues.
_HAND_EVENT_MARKER = "__lyric_hand__"

_warned_once = {"start": False, "push": False, "stop": False}


def _warn_once(key, msg):
    if not _warned_once.get(key):
        print(f"[LYRIC] {msg}")
        _warned_once[key] = True


class _LyricRequestHandler(BaseHTTPRequestHandler):
    server_version = "LyricPage/1.0"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_index()
        elif path == "/stream":
            self._serve_stream()
        elif self._is_allowed_static(path):
            self._serve_static(path)
        else:
            self.send_response(404)
            self.end_headers()

    def _is_allowed_static(self, path):
        if path in _STATIC_ALLOWED_FILES:
            return True
        return any(path.startswith(prefix) for prefix in _STATIC_ALLOWED_PREFIXES)

    def _serve_static(self, path):
        full_path = os.path.normpath(os.path.join(_PROJECT_DIR, path.lstrip("/")))
        if not full_path.startswith(_PROJECT_DIR) or not os.path.isfile(full_path):
            self.send_response(404)
            self.end_headers()
            return
        ext = os.path.splitext(full_path)[1].lower()
        content_type = _STATIC_MIME_TYPES.get(ext, "application/octet-stream")
        try:
            with open(full_path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_index(self):
        try:
            with open(_HTML_PATH, "r", encoding="utf-8") as f:
                body = f.read()
        except OSError:
            body = _FALLBACK_HTML
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_stream(self):
        server = self.server
        client_queue = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        server.register_subscriber(client_queue)

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            while server.is_running():
                try:
                    item = client_queue.get(timeout=_HEARTBEAT_SECONDS)
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == _HAND_EVENT_MARKER:
                        self.wfile.write(f"event: hand\ndata: {item[1]}\n\n".encode("utf-8"))
                    else:
                        safe_text = item.replace("\r", " ").replace("\n", " ")
                        self.wfile.write(f"data: {safe_text}\n\n".encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            server.unregister_subscriber(client_queue)


class _LyricHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subscribers = set()
        self._subscribers_lock = threading.Lock()
        self._running = True

    def register_subscriber(self, q):
        with self._subscribers_lock:
            self._subscribers.add(q)

    def unregister_subscriber(self, q):
        with self._subscribers_lock:
            self._subscribers.discard(q)

    def is_running(self):
        return self._running

    def broadcast(self, text):
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(text)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(text)
                except queue.Full:
                    pass

    def broadcast_hand(self, state):
        payload = (_HAND_EVENT_MARKER, "1" if state else "0")
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def shutdown_all(self):
        self._running = False
        with self._subscribers_lock:
            subs = list(self._subscribers)
            self._subscribers.clear()
        for q in subs:
            try:
                q.put_nowait("")
            except queue.Full:
                pass


class LyricPage:
    def __init__(self):
        self.enabled = LYRIC_ENABLE
        self._server = None
        self._thread = None
        self._started = False
        self._osc_server = None
        self._osc_thread = None
        self._last_hand_state = None
        self._hand_state_lock = threading.Lock()

    def start(self):
        if self._started:
            return
        self._started = True
        if not self.enabled:
            return
        try:
            self._server = _LyricHTTPServer((LYRIC_HOST, LYRIC_PORT), _LyricRequestHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"[LYRIC] page -> http://{LYRIC_HOST}:{LYRIC_PORT}/")
        except Exception as e:
            _warn_once("start", f"could not start page server ({e}); lyric page disabled")
            self._server = None
            return
        self._start_hand_osc()

    def _start_hand_osc(self):
        if not LYRIC_HAND_OSC_ENABLE:
            return
        if _osc_server is None or _osc_dispatcher is None:
            _warn_once(
                "hand_osc",
                f"python-osc unavailable ({_HAND_OSC_IMPORT_ERROR}); hand-state input disabled",
            )
            return
        try:
            disp = _osc_dispatcher.Dispatcher()
            disp.map(LYRIC_HAND_OSC_ADDR, self._on_hand_osc)
            self._osc_server = _osc_server.ThreadingOSCUDPServer(
                (LYRIC_HOST, LYRIC_HAND_OSC_PORT), disp
            )
            self._osc_thread = threading.Thread(target=self._osc_server.serve_forever, daemon=True)
            self._osc_thread.start()
            print(f"[LYRIC] hand-osc <- udp://{LYRIC_HOST}:{LYRIC_HAND_OSC_PORT}{LYRIC_HAND_OSC_ADDR}")
        except Exception as e:
            _warn_once("hand_osc", f"could not start hand-osc receiver ({e}); hand-state input disabled")
            self._osc_server = None

    def _on_hand_osc(self, _addr, *args):
        if not args:
            return
        try:
            value = float(args[0])
        except (TypeError, ValueError):
            return
        state = value >= 0.5
        with self._hand_state_lock:
            if state == self._last_hand_state:
                return
            self._last_hand_state = state
        if self._server is not None:
            try:
                self._server.broadcast_hand(state)
            except Exception as e:
                _warn_once("hand_osc", f"hand-state broadcast failed ({e})")

    def push(self, text, is_reassemble=False):
        if not self.enabled or self._server is None or not text:
            return
        if is_reassemble and not LYRIC_SHOW_REASSEMBLE:
            return
        try:
            self._server.broadcast(text)
        except Exception as e:
            _warn_once("push", f"push failed ({e}); lyric page disabled")

    def stop(self):
        if self._osc_server is not None:
            try:
                self._osc_server.shutdown()
                self._osc_server.server_close()
            except Exception as e:
                _warn_once("stop", f"hand-osc shutdown error ({e})")
            if self._osc_thread is not None:
                self._osc_thread.join(timeout=2.0)
            self._osc_server = None
        if self._server is None:
            return
        try:
            self._server.shutdown_all()
            self._server.shutdown()
            self._server.server_close()
        except Exception as e:
            _warn_once("stop", f"shutdown error ({e})")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None


lyric_page = LyricPage()
