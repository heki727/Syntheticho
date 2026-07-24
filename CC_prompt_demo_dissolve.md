# Claude Code 任务：给 demo_offline.py 加 OSC dissolve 发送，让离线 demo 也能驱动花朵溶解/凝聚

## 背景
- 项目路径：`/Users/heki/Desktop/final Syntheticho/final_llm`
- TouchDesigner 里花朵的凝聚/弥散由 OSC 通道 `/dissolve` 驱动（`osc_in`，监听 `127.0.0.1:10727`）。
  - TD 侧表达式：`attract_force.gain = (1 - osc_in['dissolve']) * 0.05`，`thresh2.threshold = 1 + osc_in['dissolve'] * -1.2`。
  - `dissolve = 0` → 花完全凝聚；`dissolve = 1` → 花完全弥散。
- 目前只有主程序 `esp32.py` 会发 `/dissolve`（见其第 1930-1932 行：`dissolve_amount = 1.0 - flower_val`）。
- `hand_osc.py` 只发 `/handx /handy /handon`，不发 dissolve。
- `demo_offline.py` 是离线文字 demo（无摄像头、无检测），当前**完全不发 OSC**，所以跑 demo 时花一直卡在凝聚不动。

## 目标
在 **`demo_offline.py`** 里加一个轻量 OSC 发送，按每句独白的 `stage` 给花一个 dissolve 目标值，并在该句的等待时间内**平滑过渡**（约 30fps 连续发送），让花随独白半溶解半凝聚地移动。不要引入摄像头/检测，不要改动 `esp32.py`、`hand_osc.py`、`whisper_tts.py`、`lyric_page.py`。

## 具体改动（只改 demo_offline.py）

1. 顶部导入与常量：
   ```python
   from pythonosc.udp_client import SimpleUDPClient
   OSC_IP = "127.0.0.1"
   OSC_PORT = 10727
   DISSOLVE_SEND_FPS = 30      # dissolve 平滑发送帧率
   DISSOLVE_SMOOTH = 0.03      # 每帧向目标逼近的比例(0~1，越小越慢越顺)。
                               # 放慢是刻意的：主程序 esp32.py 里 flower_val 变化极慢
                               # (40 秒溶解 / 20 秒回凝)，所以观感是缓慢漂移、几乎不到两端。
   # 全局钳制：无论目标怎么给，dissolve 都锁在中间带，保证“不完全溶解也不完全凝聚”
   DISSOLVE_MIN = 0.22
   DISSOLVE_MAX = 0.68
   ```

2. 各 stage 对应的 dissolve 目标值（0=凝聚, 1=弥散），加到文件里作为常量字典：
   ```python
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
   ```
   注：目标值全部压在 0.25–0.65 之间——这是"半溶解半凝聚"区间，花始终有明显结构又在流动，不会成一团实心花、也不会彻底散掉。

3. 在 `main()` 里：
   - 进入循环前创建 `osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)`，并初始化 `current_dissolve = DISSOLVE_DEFAULT`（起手就在中间带，不要从 0 开始）。
   - 把原本每句结束后的单次 `time.sleep(wait)`，替换成一个**按 DISSOLVE_SEND_FPS 分帧的 while 循环**：在 `wait` 秒内，每帧把 `current_dissolve` 用 EMA 朝 `STAGE_DISSOLVE.get(stage, DISSOLVE_DEFAULT)` 逼近
     （`current_dissolve += (target - current_dissolve) * DISSOLVE_SMOOTH`），
     逼近后再做全局钳制 `current_dissolve = min(DISSOLVE_MAX, max(DISSOLVE_MIN, current_dissolve))`，
     然后 `osc_client.send_message("/dissolve", float(current_dissolve))` 和
     `osc_client.send_message("/dissolve_amount", float(current_dissolve))`，
     `time.sleep(1.0 / DISSOLVE_SEND_FPS)`，直到累计时间达到 `wait`。
   - 保持原有的 `whisper_tts.speak(...)` 和 `lyric_page.push(...)` 调用与时序不变（在每句开头发送一次即可，dissolve 循环在其后运行）。

4. 不要发 `/handx /handy /handon`（那是 hand_osc.py 的职责，避免抢占/冲突）。`/confidence` 可不发。

## 约束
- 只修改 `demo_offline.py` 一个文件。
- 依赖 `python-osc`（`pythonosc`），与 `hand_osc.py` 同款，环境里应已装；若缺失请在报告里指出，不要自动改环境。
- 不引入摄像头、MediaPipe、API 调用。
- 保持 `Ctrl+C` 退出与 `finally` 里 `whisper_tts.stop()`/`lyric_page.stop()` 的清理逻辑；退出时不要发 0（会让花突然实心凝聚），保持最后一帧或发 `DISSOLVE_DEFAULT` 即可。

## 验证
1. 语法检查：`python -m py_compile demo_offline.py`。
2. 干跑说明：确认循环里 dissolve 会从上一句的值平滑过渡到当前 stage 目标值（可临时 print `current_dissolve` 每 0.5s 一次自查，验证后移除或注释）。
3. 与 TD 联调：启动 `demo_offline.py`，在 TD 的 `osc_in` CHOP 上确认 `dissolve` 通道随独白 stage 变化；`collapsing` 句时应升向 1（花弥散），`soothed/acknowledged` 句时降向 ~0.1（花凝聚）。
4. 报告：改了哪些行、STAGE_DISSOLVE 取值、是否需要手动 `pip install python-osc`。
