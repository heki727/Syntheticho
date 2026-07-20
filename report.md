1. 执行前 grep 结果

```text
718:ESP32_CONTROL_BASE_URL = esp32_control_base_url(ESP32_CAMERA_URL)
735:def set_esp32_camera_profile(index: int, reason: str):
861:if SET_CAMERA_PROFILE_ON_STARTUP:
```

确认：`set_esp32_lowlight_params` 无匹配，未发现旧定义。

2. 新增函数行号、调用行号

新增函数定义：`esp32.py:750`

startup 调用：`esp32.py:878`

3. 执行后 grep 结果，确认三个数值正确

```text
750:def set_esp32_lowlight_params():
753:        ("ae_level", 1),        # auto-exposure target +1
754:        ("gainceiling", 3),     # gain ceiling 16x
755:        ("led_intensity", 100), # onboard LED fill light
763:    print("[ESP32] low-light params applied (ae_level=1, gainceiling=3, led=100)")
878:    set_esp32_lowlight_params()
```

确认：`ae_level=1`、`gainceiling=3`、`led_intensity=100` 三个实测值均已按要求写入，startup 块已调用 `set_esp32_lowlight_params()`。
