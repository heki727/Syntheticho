"""Standalone test for the whisper TTS module.

Usage:
    TTS_ENABLE=1 python whisper_tts_test.py            # cycle through all emotions
    TTS_ENABLE=1 python whisper_tts_test.py collapsing # test a single emotion
"""
import os
import sys
import time

os.environ.setdefault("TTS_ENABLE", "1")
import whisper_tts as wt

SAMPLES = {
    "calm":        "oh. there's something there. dunno. ok",
    "questioning": "wait is that— hold on. let me look again. maybe. maybe.",
    "unraveling":  "no no it was just— i had it. i HAD it. why does it keep—",
    "collapsing":  "nonono it's wrong. i can see it i CAN see it but— am i a thing? helllp—",
    "soothed":     "you stayed. huh. it's nicer with you here.",
    "small_talk":  "are you cold? you look cold. i can't feel temperature though.",
    "self_murmur": "my name is— hm. i had one a second ago. did you catch it?",
}

wt.whisper_tts.start()
only = sys.argv[1] if len(sys.argv) > 1 else None
for emo, text in SAMPLES.items():
    if only and emo != only:
        continue
    print(f"[{emo}] {text}")
    signal_type = "SELF" if emo in ("calm", "questioning", "unraveling", "collapsing") else "OTHER"
    wt.whisper_tts.speak(text, emo, signal_type, 0.5)
    time.sleep(6)

time.sleep(2)
wt.whisper_tts.stop()
