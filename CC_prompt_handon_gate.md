# TD 任务总 Prompt — 点云 handon 门控可见性（路径 A）+ 工程现状交接

> 用于在新窗口继续。执行环境：TouchDesigner MCP，工程 `/project1`（flower&hand）。
> **重要：动手前先用 `execute_python_script` 读实时工程核对本文所列节点/数值，不要凭空假设；本文数值是交接时的状态，可能与你接手时略有出入。**

---

## 一、工程结构（已从实时工程读出，勿重新假设）

**两套独立视觉，最终在 `flower_plus_cam` 合成 → `null4` 输出：**

- **手云**（GPU 粒子）：`hc_state → hc_vel → hc_pos → hc_color` →（`geo_cloud`+`geo_web`）→ `cam_render`(相机 `cam_ortho`) → `cam_gate` → `flower_plus_cam`
- **花**（点力场模拟）：`start →（attract/noise/force 力场）→ points → null11` →（`geo1`+`geo2`）→ `render1`(相机 `cam1`) → `depth1 → lumablur1 → transform2` → `flower_plus_cam`

**OSC 输入** `/project1/osc_in`（oscinCHOP，端口 10727），7 通道：`dissolve, score, confidence, dissolve_amount, handon, handx, handy`
- `handx/handy/handon` 来自 `hand_osc.py`（MediaPipe，含 `MIRROR_X=True` 与 8 帧无手迟滞）
- `dissolve/dissolve_amount/confidence` 来自主程序 `esp32.py`（presence 状态机；`dissolve=1-flower_val`）。`demo_offline.py` 不发任何 OSC。

**手部位置链**：`osc_in → hand_in(replaceCHOP) → hand_pt(constantTOP) → 粒子仿真`
- ⚠️ `hand_pt` 通道是**交换**的：`colorr=handy, colorg=handx, colorb=handon`。下游 `hc_state`/`hc_vel` 读 `h.r=手Y, h.g=手X`。

---

## 二、本轮已做的改动（当前状态，供你了解，勿重复）

- `hc_vel_glsl`：`hx=(0.5-st.g)*2.0*ASPECT; hy=(0.5-st.b)*2.0;`（方向已校对）；`FLOWER_X=0.399, FLOWER_Y=-0.130`；`CORE_BIAS 0.7`；`DAMP 0.06`；`uTime` 已改为 `vec4` uniform，绑定在 GLSL TOP 的 Vectors 页 `vec0name=uTime`, `vec0valuex = absTime.seconds`（流动动画已生效，勿再动）。
- `hc_color_glsl`：`hx/hy` 与 hc_vel 一致；含 `GLOW_R`、`GLOW_XN` 两个自定义 define 控制发光核心。
- `hc_state_glsl`（已整段重写干净）：`TOUCH_CX=0.388, TOUCH_CY=0.565, TOUCH_R=0.05, RELEASE_FRACTION=0.5, RELEASE_SPEED=0.03, RETURN_SPEED=0.05`。触发圈已按实测花心（画面左上原点 ≈ 水平0.605/垂直0.561）反推对齐。
- `transform2`（花分支）：`rotate=-90, sx=-2, sy=2`（`sx=-2` 是为把花垂直翻转对齐手云）。
- `cam_ortho`：`tx=0, ty=0, winx=0, winy=0`（居中），`orthowidth=3.8`。
- `cam_render` 相机 = `cam_ortho`（手云保持正交）。
- 调试标记 `fx_target_marker` 已删除。

> 交互逻辑摘要：手进入 `hc_state` 的触发圈(TOUCH_CX/CY,半径 TOUCH_R) → `rel` 上升 → `hc_vel` 里 `home=mix(handHome, flowerHome, rel)` 让一半(RELEASE_FRACTION)粒子飞向花靶点。花本身的凝聚/弥散是另一套，由 `osc_in['dissolve']` 驱动，与手位置无关。

---

## 三、本任务目标：无手时点云消失（handon 门控），有手淡入、无手淡出，不能硬切

`handon` 目前全工程 **0 处被引用**，可见性从未接过——这是要补的。走**路径 A（合成层亮度闸门，推荐）**。

### 已完成（交接时）
- 新建 `/project1/handon_sel`(selectCHOP)：`osc_in → handon_sel`，`channames='handon'`，已输出单通道 `handon`。
- 新建 `/project1/handon_smooth`(lagCHOP)：`handon_sel → handon_smooth`，输出通道 `handon`。**Lag 值尚未设**。其参数名为 `lag1`(上升)、`lag2`(下降)、`lagunit`。
- 新建 `/project1/handon_const`(constantTOP)：**尚未配置、尚未接线**。

### 剩余步骤（请完成）
1. **设 Lag 平滑**：`handon_smooth` 设 `lagunit=seconds`，`lag1=0.4`（出现稍快），`lag2=0.6`（消失稍慢）。确认输出在 0↔1 间平滑过渡、无硬跳。
2. **常量亮度**：`handon_const` 的 `colorr/colorg/colorb` 全部用表达式 `op('handon_smooth')['handon']`，`alpha=1`；分辨率设为与 `render1` 一致（1280×720），或用 Common 页 Resolution=Input。
3. **闸门相乘（切点必须在 feedback 之前的 `multiply1`）**：`multiply1`(multiply TOP) 当前输入为 `math2`、`thresh2`；把 `handon_const` 接为它的**第 3 个输入**，使 `multiply1` 输出 = 原亮度 × handon_smooth。
   - 效果：无手→亮度归零、点云消失，拖尾经 `feedback1` 自然衰减无残影；有手→恢复。
   - ⚠️ 不要切在 `add1` 之后，否则 `feedback1` 会残留拖尾闪烁。

---

## 四、严禁改动
- 手部位置链 `hand_in / hand_pt / handx / handy` —— 不动。
- 有手(handon=1)状态下点云的外观、颜色、密度、拖尾、TOUCH/FLOWER 等 —— 不动。
- `osc_in` 端口/通道、`dissolve/score/confidence` 现有用途 —— 不动。
- `render1 / cam1 / geo1 / geo2 / 材质 pbr1 / transform2` —— 不动。
- 路径 A 全程**不碰任何 GLSL**，只在 `multiply1` 一路加一个相乘输入。

---

## 五、完成后报告（verification）
1. `handon_sel` / `handon_smooth` 节点路径与 lag 值（lag1/lag2/lagunit）。
2. `multiply1` 链新增相乘的接法（贴出 `multiply1.inputs` 三个输入路径），确认切点在 feedback 之前。
3. 确认 `handon` 现在被引用（0 → N 处），列出引用位置（`handon_sel` 源、`handon_const` 表达式等）。
4. 测试：把 `osc_in['handon']`（或断开 hand_osc）在 1↔0 间切换，确认无手时 `null4` 点云完全不可见且拖尾无残留、有手时外观与改动前一致、淡入淡出平滑（可用 `get_top_image` 抓 `null4` 前后对比，或读 `multiply1` 输出亮度均值随 handon 变化）。
5. 确认 `hand_pt / handx / handy / render1 / geo / 有手外观` 未被改动。

## 六、可选（路径 B，路径 A 验证跨屏同步后再考虑）
无手时让粒子向下溶解坠落而非原地淡出：在 `hc_vel_glsl` 加 uniform `uPresence=handon_smooth`，`uPresence<1` 时叠加向下力+散开；`hc_color_glsl` 用 `uPresence` 压 alpha。**必须保证 `uPresence=1` 时与现状零行为差异**。
