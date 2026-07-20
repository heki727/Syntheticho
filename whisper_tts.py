"""whisper_tts.py — Syntheticho local whisper TTS module (offline).

Concept: an unsourced synthetic voice. Low volume, breathy, mumbled,
fragmented — imperfection is a feature, not a bug.

This module is a self-contained side-channel: it queues monologue text
for background synthesis/playback and never blocks the caller. It does
not touch presence / emotion / OSC / TouchDesigner / LLM generation logic.
"""

import os
import re
import time
import queue
import random
import shutil
import threading
import subprocess

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    from scipy import signal as scipy_signal
    _AUDIO_OK = True
    _AUDIO_IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - environment dependent
    _AUDIO_OK = False
    _AUDIO_IMPORT_ERR = _e

# ======================= 可调参数（现场调音色就改这里）=======================
TTS_ENABLE = os.environ.get("TTS_ENABLE", "1") == "1"   # 总开关，默认开
SPEAK_REASSEMBLE = os.environ.get("TTS_SPEAK_REASSEMBLE", "0") == "1"  # 是否朗读机械读数
OUTPUT_DEVICE = os.environ.get("TTS_OUTPUT_DEVICE", "")  # 空=默认设备；可填 sounddevice 设备名/索引
CAT_BRACKETS_ONLY = os.environ.get("TTS_CAT_BRACKETS_ONLY", "1") == "1"  # cat 人格是否只念括号内文本

QUEUE_MAX = 2               # 积压超过就丢最旧（现场不要越念越滞后）
INTERRUPT_ON_NEW = True     # 新独白到来是否打断当前在念的

# —— 低语基础音色（所有情绪共享的底色）——
BASE_GAIN_DB      = -14.0   # 整体压低音量
LOWPASS_HZ        = 150     # 低通截止，砍掉共振峰只留基频区，人声→低频嗡鸣
BREATH_MIX        = 0.10    # 混入的气声/白噪比例（0~0.3）
REVERB_MIX        = 0.12    # 轻混响 wet 比例（0~0.4）
REVERB_DECAY      = 0.28    # 简易反馈延迟的衰减
HP_HZ             = 120     # 高通去掉低频轰鸣，更像气声
ENVELOPE_FOLLOW   = 0.85    # 包络保留强度 0~1：嗡鸣跟随原始语音音节起伏的程度

# —— 情绪 → 语速/音高/停顿 映射（8 个 stage，改这里最直接）——
# rate: >1 更慢（Piper length_scale 语义；这里统一成"越大越慢"），
# pitch: 半音，正=升；pause: 句间随机停顿秒 (min,max)；frag: 碎片化断句概率
EMOTION_VOICE = {
    "calm":         dict(rate=1.25, pitch=-2.0, pause=(0.5, 1.1), frag=0.15, gain_db=-15),
    "soothed":      dict(rate=1.20, pitch=-1.0, pause=(0.4, 1.0), frag=0.15, gain_db=-14),
    "acknowledged": dict(rate=1.15, pitch=-1.0, pause=(0.4, 0.9), frag=0.15, gain_db=-14),
    "small_talk":   dict(rate=1.05, pitch=+0.5, pause=(0.3, 0.8), frag=0.20, gain_db=-13),
    "questioning":  dict(rate=1.00, pitch=+1.0, pause=(0.3, 0.7), frag=0.30, gain_db=-13),
    "self_murmur":  dict(rate=1.15, pitch=-1.5, pause=(0.5, 1.2), frag=0.35, gain_db=-15),
    "unraveling":   dict(rate=0.92, pitch=+2.0, pause=(0.15, 0.5), frag=0.55, gain_db=-12),
    "collapsing":   dict(rate=0.85, pitch=+3.0, pause=(0.05, 0.35), frag=0.70, gain_db=-11),
    # shock 可选：并入 collapsing，或单列（当前单列，参数比 collapsing 更极端）
    "shock":        dict(rate=0.80, pitch=+3.5, pause=(0.05, 0.3), frag=0.75, gain_db=-11),
}
DEFAULT_VOICE = EMOTION_VOICE["calm"]

# —— Piper 配置 ——
PIPER_MODEL = os.environ.get("PIPER_MODEL", "voices/en_US-lessac-low.onnx")
PIPER_BIN   = os.environ.get("PIPER_BIN", "piper")   # 二进制名或路径
SAMPLE_RATE = 16000  # 与所选 voice 一致
# ==========================================================================

_warned_once = {"engine": False, "audio": False}


def _warn_once(key, msg):
    if not _warned_once.get(key):
        print(f"[TTS] {msg}")
        _warned_once[key] = True


# --------------------------------------------------------------------------
# 引擎后端
# --------------------------------------------------------------------------

class PiperBackend:
    """Offline Piper TTS backend. Tries the python package first, falls
    back to the `piper` CLI binary via subprocess."""

    def __init__(self, model_path=PIPER_MODEL, piper_bin=PIPER_BIN):
        self.model_path = model_path
        self.piper_bin = piper_bin
        self._py_voice = None
        self._mode = None  # "python" | "binary" | None

        if os.path.isfile(self.model_path):
            try:
                from piper import PiperVoice
                self._py_voice = PiperVoice.load(self.model_path)
                self._mode = "python"
            except Exception:
                self._py_voice = None

        if self._mode is None and shutil.which(self.piper_bin) and os.path.isfile(self.model_path):
            self._mode = "binary"

    def available(self):
        return self._mode is not None

    def synth(self, text, rate):
        """Returns (np.float32 mono signal, sample_rate). Raises on failure."""
        if self._mode == "python":
            return self._synth_python(text, rate)
        elif self._mode == "binary":
            return self._synth_binary(text, rate)
        raise RuntimeError("Piper backend not available (missing model or binary)")

    def _synth_python(self, text, rate):
        import io
        import wave
        from piper.config import SynthesisConfig
        syn_config = SynthesisConfig(length_scale=rate)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._py_voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            sr = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)
            sampwidth = wav_file.getsampwidth()
        if sampwidth == 2:
            sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            sig = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        return sig, sr

    def _synth_binary(self, text, rate):
        proc = subprocess.run(
            [self.piper_bin, "--model", self.model_path, "--length_scale", str(rate),
             "--output_file", "-"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"piper binary failed: {proc.stderr.decode(errors='ignore')[:200]}")
        import io
        data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr


class CoquiBackend:
    """Placeholder for a Coqui TTS backend, kept unimplemented on purpose —
    Piper is the default (offline-friendly, fast cold start, no torch dep).

    To implement: pip install TTS; load a model with TTS.api.TTS(...);
    call .tts(text) to get a numpy waveform + sample rate, then return it
    in the same (np.float32 mono, sr) shape that synth() returns above.
    """

    def available(self):
        return False

    def synth(self, text, rate):
        raise NotImplementedError("CoquiBackend is a stub; use PiperBackend")


# --------------------------------------------------------------------------
# 音频后处理（都作用在 float32 mono 上）
# --------------------------------------------------------------------------

def apply_highpass(sig, sr, hz):
    if hz <= 0 or len(sig) < 16:
        return sig
    b, a = scipy_signal.butter(2, hz / (sr / 2), btype="high")
    return scipy_signal.filtfilt(b, a, sig).astype(np.float32)


def apply_lowpass(sig, sr, hz):
    if hz <= 0 or hz >= sr / 2 or len(sig) < 16:
        return sig
    b, a = scipy_signal.butter(4, hz / (sr / 2), btype="low")
    return scipy_signal.filtfilt(b, a, sig).astype(np.float32)


def apply_pitch(sig, sr, semitones):
    """Lightweight pitch shift via resample-and-restretch (no librosa/torch)."""
    if abs(semitones) < 1e-3 or len(sig) < 16:
        return sig
    factor = 2.0 ** (semitones / 12.0)
    n_stretched = max(1, int(len(sig) / factor))
    stretched = scipy_signal.resample(sig, n_stretched)
    restored = scipy_signal.resample(stretched, len(sig))
    return restored.astype(np.float32)


def extract_envelope(sig, sr):
    """从原始语音提取平滑的 RMS 振幅包络（音节级强弱轮廓）。

    必须在低通之前对原始语音调用——低通之后音节强弱信息已被砍掉。
    返回与 sig 等长、范围约 0~1 的包络数组。
    """
    if len(sig) < 16:
        return np.ones_like(sig)
    rectified = np.abs(sig)
    # 平滑窗约 20ms，对应音节级别的强弱起伏（不是逐样本、也不是整句）
    win = max(1, int(sr * 0.02))
    if win > 1:
        kernel = np.ones(win) / win
        env = np.convolve(rectified, kernel, mode="same")
    else:
        env = rectified
    peak = np.max(env) if len(env) else 0.0
    if peak > 1e-9:
        env = env / peak
    return env.astype(np.float32)


def apply_envelope(sig, envelope, follow):
    """把原始语音的包络乘回（已低通的）嗡鸣信号，实现音节级同步。

    follow=0 时不改变 sig；follow=1 时嗡鸣完全跟随包络起伏。
    中间值按 (1-follow) + follow*envelope 的增益曲线混合，
    保留少量底噪避免音节间隙彻底断掉。
    """
    if follow <= 0 or len(sig) < 16:
        return sig
    if len(envelope) != len(sig):
        # 长度不一致（理论上不会发生）时安全退回，不做包络
        return sig
    gain = (1.0 - follow) + follow * envelope
    return (sig * gain).astype(np.float32)


def add_breath(sig, level):
    if level <= 0 or len(sig) < 16:
        return sig
    envelope = np.abs(sig)
    win = max(1, int(len(sig) / 200))
    if win > 1:
        kernel = np.ones(win) / win
        envelope = np.convolve(envelope, kernel, mode="same")
    noise = np.random.default_rng().normal(0, 1, len(sig)).astype(np.float32)
    breathy = noise * envelope * level
    return (sig + breathy).astype(np.float32)


def add_reverb(sig, sr, mix, decay):
    if mix <= 0 or len(sig) < 16:
        return sig
    out = sig.copy()
    for tap_ms, tap_gain in ((23, 0.5), (47, 0.35), (79, 0.22)):
        delay = int(sr * tap_ms / 1000.0)
        if delay <= 0 or delay >= len(sig):
            continue
        delayed = np.zeros_like(sig)
        delayed[delay:] = sig[:-delay] * tap_gain * decay
        out = out + delayed
    return ((1 - mix) * sig + mix * out).astype(np.float32)


def apply_gain_db(sig, db):
    return (sig * (10.0 ** (db / 20.0))).astype(np.float32)


def _soft_limit(sig, ceiling=0.95):
    peak = np.max(np.abs(sig)) if len(sig) else 0.0
    if peak > ceiling:
        sig = sig * (ceiling / peak)
    return sig.astype(np.float32)


def whisperize(sig, sr, voice_params):
    sig = apply_highpass(sig, sr, HP_HZ)
    sig = apply_pitch(sig, sr, voice_params.get("pitch", 0.0))
    # 包络必须在低通之前从（含音节强弱的）语音提取
    envelope = extract_envelope(sig, sr)
    sig = apply_lowpass(sig, sr, LOWPASS_HZ)
    # 低通后信号已是低频嗡；把原语音包络乘回，让嗡随音节起伏（音节级同步）
    sig = apply_envelope(sig, envelope, ENVELOPE_FOLLOW)
    sig = add_breath(sig, BREATH_MIX)
    sig = add_reverb(sig, sr, REVERB_MIX, REVERB_DECAY)
    sig = apply_gain_db(sig, BASE_GAIN_DB + voice_params.get("gain_db", 0.0) - DEFAULT_VOICE["gain_db"])
    sig = _soft_limit(sig)
    return sig


# --------------------------------------------------------------------------
# 文本切片
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"(?<=[.,!?—])\s+|\.\.\.\s*")


def split_fragments(text, frag_prob):
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    if not parts:
        parts = [text]

    fragments = []
    rng = random.Random()
    for part in parts:
        words = part.split()
        if rng.random() < frag_prob and len(words) > 3:
            # further break this chunk into 2 pieces at a random point
            cut = rng.randint(1, len(words) - 1)
            fragments.append(" ".join(words[:cut]))
            fragments.append(" ".join(words[cut:]))
        else:
            fragments.append(part)

    result = []
    for frag in fragments:
        pause = rng.uniform(0.05, 0.2) if rng.random() < frag_prob else 0.0
        result.append((frag, pause))
    return result


def extract_cat_bracket_text(thought):
    match = re.search(r"\((.*)\)", thought, flags=re.S)
    return match.group(1).strip() if match else thought


# --------------------------------------------------------------------------
# 主类
# --------------------------------------------------------------------------

class WhisperTTS:
    def __init__(self):
        self.enabled = TTS_ENABLE
        self._queue = queue.Queue(maxsize=QUEUE_MAX)
        self._thread = None
        self._running = False
        self._stop_current = threading.Event()
        self._backend = None
        self._started = False

    def _ensure_backend(self):
        if self._backend is not None:
            return
        self._backend = PiperBackend()

    def start(self):
        if self._started:
            return
        self._started = True
        if not _AUDIO_OK:
            _warn_once("audio", f"audio stack unavailable ({_AUDIO_IMPORT_ERR}); whisper TTS disabled")
            return
        self._ensure_backend()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text, emotion, signal_type, coherence):
        if not self.enabled or not _AUDIO_OK or not text:
            return
        if not self._running:
            return

        if CAT_BRACKETS_ONLY and "(" in text and ")" in text:
            text = extract_cat_bracket_text(text)
        if not text.strip():
            return

        job = (text, emotion, signal_type, coherence)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                pass

        if INTERRUPT_ON_NEW:
            self._stop_current.set()

    def _worker(self):
        while self._running:
            try:
                text, emotion, signal_type, coherence = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._stop_current.clear()
            voice = EMOTION_VOICE.get(emotion, DEFAULT_VOICE)

            try:
                if INTERRUPT_ON_NEW:
                    sd.stop()
                for frag_text, pause_after in split_fragments(text, voice["frag"]):
                    if self._stop_current.is_set() or not self._running:
                        break
                    try:
                        sig, sr = self._backend.synth(frag_text, voice["rate"])
                    except Exception as e:
                        _warn_once("engine", f"synthesis unavailable ({e}); whisper TTS disabled")
                        self._running = False
                        break
                    sig = whisperize(sig, sr, voice)
                    if self._stop_current.is_set() or not self._running:
                        break
                    sd.play(sig, sr, device=(OUTPUT_DEVICE or None))
                    sd.wait()
                    if pause_after > 0:
                        time.sleep(pause_after)
            except Exception as e:
                _warn_once("engine", f"playback error ({e}); whisper TTS disabled")
                self._running = False

    def stop(self):
        self._running = False
        if _AUDIO_OK:
            try:
                sd.stop()
            except Exception:
                pass
        self._stop_current.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


whisper_tts = WhisperTTS()
