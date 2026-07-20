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

Calibration constants (top of file):
    CAMERA_INDEX        - cv2.VideoCapture index for the built-in camera.
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

import sys
import time

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

# ======================= 可调常量（现场校准就改这里）=======================
CAMERA_INDEX = 0

POINT_LANDMARK = 8  # index fingertip. Palm-center alternative: average
                     # landmarks [0, 5, 9, 13, 17] instead of taking one.

MIRROR_X = True   # selfie camera → mirror x so audience left/right matches
SMOOTH = 0.5       # EMA smoothing, 0..1 (0 = off, higher = smoother/laggier)

OSC_IP = "127.0.0.1"
OSC_PORT = 10727
FPS = 30

SHOW_WINDOW = False
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


def main():
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            f"[hand_osc] ERROR: could not open camera index {CAMERA_INDEX}.\n"
            f"  Possible causes:\n"
            f"  - Another process (e.g. TouchDesigner's live_in TOP) is holding it "
            f"exclusively — macOS usually allows shared access, but check TD.\n"
            f"  - Terminal/Python is not authorized: System Settings → Privacy & "
            f"Security → Camera → enable your terminal app.\n"
            f"  - CAMERA_INDEX is wrong for this machine; try 1, 2, ..."
        )
        sys.exit(1)

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

    print(f"[hand_osc] streaming /handx /handy /handon → {OSC_IP}:{OSC_PORT}")
    print("[hand_osc] Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                hand_on = 0.0
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    raw_x, raw_y = get_point_xy(hand_landmarks)
                    if MIRROR_X:
                        raw_x = 1.0 - raw_x

                    if smoothed_x is None:
                        smoothed_x, smoothed_y = raw_x, raw_y
                    else:
                        smoothed_x = SMOOTH * smoothed_x + (1.0 - SMOOTH) * raw_x
                        smoothed_y = SMOOTH * smoothed_y + (1.0 - SMOOTH) * raw_y

                    hand_on = 1.0

                    if SHOW_WINDOW:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                # else: hand not detected this frame — keep last smoothed_x/y
                # (they simply aren't updated), only hand_on drops to 0.0.

                if smoothed_x is not None:
                    osc_client.send_message("/handx", float(smoothed_x))
                    osc_client.send_message("/handy", float(smoothed_y))
                osc_client.send_message("/handon", float(hand_on))

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
