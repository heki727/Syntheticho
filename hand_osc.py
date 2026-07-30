"""hand_osc.py — MacBook camera hand tracking → OSC → TouchDesigner.

Tracks one hand via the MacBook's built-in webcam using MediaPipe Hands,
and streams its position over OSC to TouchDesigner's OSC In CHOP
(/project1/osc_in, listening on 127.0.0.1:10727). TD turns /handx,/handy
into a glowing point-cloud at the hand's position and uses /handon to
decide whether the point cloud is touching the narcissus flower.

Run:
    source ~/yolo_env/bin/activate
    cd "/Users/heki/Desktop/final Syntheticho/final_llm"
    python hand_osc.py

Calibration con
stants (top of file):
    CAMERA_INDEX        - cv2.VideoCapture index. -1 = auto-probe for an
                          external USB camera (skips index 0, the built-in
                          webcam); set to a specific number to override.
    POINT_LANDMARK       - which MediaPipe hand landmark to use as "the"
                            hand position (default 8 = index fingertip).
                            To use the palm center instead, average a few
                            landmarks (0=wrist, 5,9,13,17=finger MCPs) —
                            see `get_point_xy()` for the swap-in snippet.
    MIRROR_X             - flip x so a selfie-view camera matches what the
                            audience sees (their right hand moves the dot
                            right). Flip this if left/right feels reversed
                            on site.
    SMOOTH               - EMA smoothing factor in [0,1); 0 = no smoothing,
                            closer to 1 = heavier smoothing/more lag.
    OSC_IP / OSC_PORT    - TouchDesigner's OSC In CHOP address.
    FPS                  - target capture/send rate.
    SHOW_WINDOW          - debug preview window with landmarks drawn;
                            keep False for unattended/on-site running.
"""

import threading
import time

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

# ======================= 可调常量（现场校准就改这里）=======================
# 内置摄像头通常是 index 0，外接 UVC（RYS-AB4 等）通常是 1 或更大。
# 现场如果自动探测选错了 index，把 CAMERA_INDEX 改成实测的具体数字即可。
CAMERA_INDEX = 0          # -1 = 自动探测外接 USB 摄像头；设为具体数字则强制使用该 index
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_USE_MJPG = True     # UVC 摄像头走 MJPEG 才能到 720p30
CAMERA_PREFER_EXTERNAL = True   # 自动探测时优先选非 0 的 index（0 通常是内置摄像头）

POINT_LANDMARK = 8  # index fingertip. Palm-center alternative: average
                     # landmarks [0, 5, 9, 13, 17] instead of taking one.

MIRROR_X = True   # selfie camera → mirror x so audience left/right matches
MIRROR_Y = False   # TD 里花的旋转改变后若点云上下颠倒，改为 True
SMOOTH = 0.5       # EMA smoothing, 0..1 (0 = off, higher = smoother/laggier)

HAND_OFF_FRAMES = 8  # consecutive no-hand frames before hand_on latches to 0.0
                      # (~0.27s @ 30fps) — absorbs single-frame MediaPipe dropouts
                      # so /handon doesn't flicker 0/1 on brief misdetections.

OSC_IP = "127.0.0.1"
OSC_PORT = 10727
FPS = 30

LYRIC_OSC_IP = "127.0.0.1"
LYRIC_OSC_PORT = 10728
LYRIC_OSC_ENABLE = True  # also mirror /handon to the lyric page; flip off on-site if needed

# ===== ToF 门控 / narcissus_main.py 联动 =====
TOF_GATE_ENABLE = False   # False = 手追完全独立，忽略 /tofon；接上 ToF 硬件后改 True
ESP_FORWARD_ENABLE = True
ESP_OSC_IP = "127.0.0.1"
ESP_OSC_PORT = 10729      # narcissus_main.py 在此端口接收 /handon（参与 presence 判定）
TOF_LISTEN_PORT = 10730   # narcissus_main.py 在此端口广播 /tofon（ToF 有人=1 / 没人=0）
TOF_GATE_TIMEOUT = 5.0    # 超过此秒数没收到 /tofon → 认为 narcissus_main.py 未运行，门保持打开

SHOW_WINDOW = False
DEBUG_PRINT_POS = True   # print live hand x/y/on to the terminal
DEBUG_PRINT_INTERVAL = 0.2  # seconds between prints, so it doesn't flood at 30fps
# ==========================================================================

MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5


def get_point_xy(hand_landmarks):
    """Returns normalized (x, y) in [0,1] for the tracked point.

    Default: single landmark (index fingertip). To switch to a palm-center
    estimate, replace the body with:

        ids = [0, 5, 9, 13, 17]
        xs = [hand_landmarks.landmark[i].x for i in ids]
        ys = [hand_landmarks.landmark[i].y for i in ids]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    """
    lm = hand_landmarks.landmark[POINT_LANDMARK]
    return lm.x, lm.y


def probe_camera_index():
    """Scan indices 0..4 via AVFoundation and pick a working camera.

    Prefers index >= 1 (external USB) when CAMERA_PREFER_EXTERNAL is set,
    since index 0 is usually the built-in webcam.
    """
    candidates = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        ok = False
        w = h = 0
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
            else:
                ok = False
        cap.release()
        print(f"[hand_osc] probe index={i} ok={ok} {w}x{h}")
        candidates.append((i, ok, w, h))

    working = [c for c in candidates if c[1]]

    chosen = 0
    if CAMERA_PREFER_EXTERNAL:
        external = [c for c in working if c[0] >= 1]
        if external:
            chosen = external[0][0]
        elif working:
            chosen = working[0][0]
    elif working:
        chosen = working[0][0]

    print(f"[hand_osc] selected camera index={chosen}")
    return chosen


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if CAMERA_USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[hand_osc] camera index={index} opened at {actual_w:.0f}x{actual_h:.0f}@{actual_fps:.0f}fps")
    return cap


def resolve_camera_index():
    return CAMERA_INDEX if CAMERA_INDEX >= 0 else probe_camera_index()


def main():
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
    lyric_client = SimpleUDPClient(LYRIC_OSC_IP, LYRIC_OSC_PORT) if LYRIC_OSC_ENABLE else None
    lyric_warned = False

    # ToF 门控状态：narcissus_main.py 广播 /tofon；收不到时默认打开（可独立运行）
    tof_state = {"on": True, "at": 0.0}

    def _on_tofon(addr, *args):
        try:
            v = float(args[0]) if args else 1.0
        except (TypeError, ValueError):
            v = 1.0
        tof_state["on"] = v >= 0.5
        tof_state["at"] = time.time()

    if TOF_GATE_ENABLE:
        _tof_dispatcher = Dispatcher()
        _tof_dispatcher.map("/tofon", _on_tofon)
        try:
            _tof_server = ThreadingOSCUDPServer(("127.0.0.1", TOF_LISTEN_PORT), _tof_dispatcher)
            threading.Thread(target=_tof_server.serve_forever, daemon=True).start()
            print(f"[hand_osc] listening for /tofon on 127.0.0.1:{TOF_LISTEN_PORT}")
        except OSError as e:
            print(f"[hand_osc] tof listener failed to bind :{TOF_LISTEN_PORT} ({e}); gate stays open")
    else:
        print("[hand_osc] ToF gate disabled — hand tracking runs independently")
    esp_client = SimpleUDPClient(ESP_OSC_IP, ESP_OSC_PORT) if ESP_FORWARD_ENABLE else None

    cam_index = resolve_camera_index()
    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(
            f"[hand_osc] ERROR: could not open camera index {cam_index}.\n"
            f"  Possible causes:\n"
            f"  - Another process (e.g. TouchDesigner's live_in TOP) is holding it "
            f"exclusively — macOS usually allows shared access, but check TD.\n"
            f"  - Terminal/Python is not authorized: System Settings → Privacy & "
            f"Security → Camera → enable your terminal app.\n"
            f"  - CAMERA_INDEX is wrong for this machine; try 1, 2, ..."
        )
        while not cap.isOpened():
            time.sleep(2.0)
            cam_index = resolve_camera_index()
            cap = open_camera(cam_index)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    mp_drawing = mp.solutions.drawing_utils if SHOW_WINDOW else None

    frame_interval = 1.0 / FPS
    smoothed_x = None
    smoothed_y = None
    hand_off_streak = HAND_OFF_FRAMES  # start "off" so no false hand-on at launch
    read_fail_streak = 0  # consecutive cap.read() failures; triggers reconnect at 30 (~1s @30fps)
    last_debug_print_at = 0.0

    print(f"[hand_osc] streaming /handx /handy /handon → {OSC_IP}:{OSC_PORT}")
    print("[hand_osc] Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok or frame is None:
                read_fail_streak += 1
                if read_fail_streak == 30:
                    print("[hand_osc] camera read failed, reopening")
                if read_fail_streak >= 30:
                    cap.release()
                    cam_index = resolve_camera_index()
                    cap = open_camera(cam_index)
                    while not cap.isOpened():
                        time.sleep(2.0)
                        cam_index = resolve_camera_index()
                        cap = open_camera(cam_index)
                    read_fail_streak = 0
                else:
                    time.sleep(0.01)
                continue

            read_fail_streak = 0

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    raw_x, raw_y = get_point_xy(hand_landmarks)
                    if MIRROR_X:
                        raw_x = 1.0 - raw_x
                    if MIRROR_Y:
                        raw_y = 1.0 - raw_y

                    if smoothed_x is None:
                        smoothed_x, smoothed_y = raw_x, raw_y
                    else:
                        smoothed_x = SMOOTH * smoothed_x + (1.0 - SMOOTH) * raw_x
                        smoothed_y = SMOOTH * smoothed_y + (1.0 - SMOOTH) * raw_y

                    hand_off_streak = 0
                    hand_on = 1.0

                    if SHOW_WINDOW:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                else:
                    # hand not detected this frame — keep last smoothed_x/y
                    # (they simply aren't updated). Only latch hand_on to 0.0
                    # after HAND_OFF_FRAMES consecutive misses, so a single
                    # dropped frame doesn't flip /handon back and forth.
                    hand_off_streak += 1
                    hand_on = 0.0 if hand_off_streak >= HAND_OFF_FRAMES else 1.0

                # ToF 门控：narcissus_main.py 判"没人"时强制 handon=0，坐标也不发。
                # 收不到 /tofon（narcissus_main.py 未运行）超过 TOF_GATE_TIMEOUT → 门保持打开，独立运行。
                if TOF_GATE_ENABLE:
                    tof_gate_open = tof_state["on"] or (time.time() - tof_state["at"]) > TOF_GATE_TIMEOUT
                else:
                    tof_gate_open = True
                if not tof_gate_open:
                    hand_on = 0.0
                if tof_gate_open and smoothed_x is not None:
                    osc_client.send_message("/handx", float(smoothed_x))
                    osc_client.send_message("/handy", float(smoothed_y))
                osc_client.send_message("/handon", float(hand_on))
                if esp_client is not None:
                    try:
                        esp_client.send_message("/handon", float(hand_on))
                    except Exception:
                        pass
                if lyric_client is not None:
                    try:
                        lyric_client.send_message("/handon", float(hand_on))
                    except Exception as e:
                        if not lyric_warned:
                            print(f"[hand_osc] lyric-page osc send failed (continuing): {e}")
                            lyric_warned = True

                if DEBUG_PRINT_POS and loop_start - last_debug_print_at >= DEBUG_PRINT_INTERVAL:
                    px = smoothed_x if smoothed_x is not None else 0.0
                    py = smoothed_y if smoothed_y is not None else 0.0
                    # print(f"[hand_osc] x={px:.3f} y={py:.3f} on={hand_on:.0f}")
                    last_debug_print_at = loop_start

                if SHOW_WINDOW:
                    label = f"x={smoothed_x if smoothed_x is not None else 0:.3f} " \
                            f"y={smoothed_y if smoothed_y is not None else 0:.3f} " \
                            f"on={hand_on:.0f}"
                    cv2.putText(
                        frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2,
                    )
                    cv2.imshow("hand_osc debug", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break

            except Exception as e:
                print(f"[hand_osc] frame error (continuing): {e}")

            elapsed = time.time() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[hand_osc] Ctrl+C received, shutting down")
    finally:
        hands.close()
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("[hand_osc] camera released, exiting")


if __name__ == "__main__":
    main()
