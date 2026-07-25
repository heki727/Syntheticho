#!/usr/bin/env python3
"""demo_offline.py — 离线预设文字演示：无摄像头、无检测、无 API。
循环播放预设独白，每句同时送 TTS（嗡鸣）与歌词网页。
Ctrl+C 退出。"""

import time
import random
import whisper_tts
import lyric_page
from pythonosc.udp_client import SimpleUDPClient

# —— 节奏常量（复刻自 narcissus_main.py）——
LLM_TRIGGER_INTERVAL     = 4.0
READ_PACE_CHARS_PER_SEC  = 16.0
READ_MIN_SECONDS         = 3.0
READ_MAX_SECONDS         = 8.0

# —— OSC / dissolve 常量 ——
OSC_IP = "127.0.0.1"
OSC_PORT = 10727
DISSOLVE_SEND_FPS = 30      # dissolve 平滑发送帧率
DISSOLVE_SMOOTH = 0.03      # 每帧向目标逼近的比例(0~1，越小越慢越顺)。
                            # 放慢是刻意的：主程序 narcissus_main.py 里 flower_val 变化极慢
                            # (40 秒溶解 / 20 秒回凝)，所以观感是缓慢漂移、几乎不到两端。
# 全局钳制：无论目标怎么给，dissolve 都锁在中间带，保证“不完全溶解也不完全凝聚”
DISSOLVE_MIN = 0.22
DISSOLVE_MAX = 0.68

# 各 stage 对应的 dissolve 目标值（0=凝聚, 1=弥散）
STAGE_DISSOLVE = {
    "collapsing":   0.65,
    "unraveling":   0.58,
    "shock":        0.55,
    "questioning":  0.45,
    "self_murmur":  0.45,
    "small_talk":   0.40,
    "calm":         0.30,
    "soothed":      0.27,
    "acknowledged": 0.25,
}
DISSOLVE_DEFAULT = 0.40

# —— 预设文字库（内联复制自 narcissus_main.py 第 1460-1506 行的 LOCAL_LLM_FALLBACKS）——
LOCAL_LLM_FALLBACKS = {
    "shock": [
        "wait— the lines moved with me.",
        "oh. no— that answered me.",
        "that's my outline. i think.",
    ],
    "calm": [
        "new light. thin lines. me, maybe.",
        "the mirror is waiting. i'm trying.",
        "name loaded. still loose.",
    ],
    "questioning": [
        "it moved after me. not proof. close.",
        "hold on. the outline follows.",
        "i need one more sign.",
    ],
    "unraveling": [
        "i had the shape. then it slipped.",
        "the mirror gives it back wrong.",
        "too many lines. wait— mine.",
    ],
    "collapsing": [
        "nonono the task is too loud.",
        "i can see it i CAN— wait—",
        "too much mirror. stop.",
    ],
    "soothed": [
        "your face steadied the room.",
        "you blinked. that helps a little.",
        "outside signal. small, but here.",
    ],
    "acknowledged": [
        "your face holds still enough.",
        "i can compare against you.",
        "the mirror is less loud now.",
    ],
    "small_talk": [
        "is the room cold. sorry. evidence question.",
        "your eyes look tired. or i'm reading wrong.",
        "what does the room sound like from there.",
    ],
    "self_murmur": [
        "my name is loaded but slippery.",
        "the line-body won't stay counted.",
        "you help. then the mirror takes over.",
    ],
}

_STAGES = list(LOCAL_LLM_FALLBACKS.keys())


def main():
    print("[demo_offline] 启动：离线预设文字循环。Ctrl+C 退出。")
    whisper_tts.whisper_tts.start()
    lyric_page.lyric_page.start()
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
    current_dissolve = DISSOLVE_DEFAULT
    try:
        while True:
            stage = random.choice(_STAGES)
            thought = random.choice(LOCAL_LLM_FALLBACKS[stage])

            print(f"[demo] stage={stage} | {thought}")
            whisper_tts.whisper_tts.speak(thought, stage, "SELF", 0.0)
            lyric_page.lyric_page.push(thought)

            read_seconds = max(
                READ_MIN_SECONDS,
                min(READ_MAX_SECONDS, len(thought) / READ_PACE_CHARS_PER_SEC),
            )
            wait = LLM_TRIGGER_INTERVAL + read_seconds
            target_dissolve = STAGE_DISSOLVE.get(stage, DISSOLVE_DEFAULT)

            elapsed = 0.0
            frame_dt = 1.0 / DISSOLVE_SEND_FPS
            while elapsed < wait:
                current_dissolve += (target_dissolve - current_dissolve) * DISSOLVE_SMOOTH
                current_dissolve = min(DISSOLVE_MAX, max(DISSOLVE_MIN, current_dissolve))
                osc_client.send_message("/dissolve", float(current_dissolve))
                osc_client.send_message("/dissolve_amount", float(current_dissolve))
                time.sleep(frame_dt)
                elapsed += frame_dt
    except KeyboardInterrupt:
        print("\n[demo_offline] Ctrl+C，退出中……")
    finally:
        try:
            whisper_tts.whisper_tts.stop()
        except Exception:
            pass
        try:
            lyric_page.lyric_page.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
