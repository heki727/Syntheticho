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

# ======================= 可调参数 =======================
LYRIC_ENABLE = os.environ.get("LYRIC_ENABLE", "1") == "1"   # 总开关，默认开
LYRIC_PORT = int(os.environ.get("LYRIC_PAGE_PORT", "8137"))
LYRIC_HOST = os.environ.get("LYRIC_PAGE_HOST", "127.0.0.1")
LYRIC_SHOW_REASSEMBLE = os.environ.get("LYRIC_SHOW_REASSEMBLE", "0") == "1"  # 是否显示 reunite 阶段的机械读数句
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
                    text = client_queue.get(timeout=_HEARTBEAT_SECONDS)
                    safe_text = text.replace("\r", " ").replace("\n", " ")
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
