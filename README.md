# tomato_grasp

单次垂直抓取。复用 visual_servo 的 红色 HSV 定位（相机为高位臂的 D435i）和手眼矩阵，复用 piper_ws 的 ROSArm 状态确认与夹爪反馈，不启动 SAM/YOLO 或新驱动。

流程：到观察位并确认到位 → 打开 RGB-D、锁定唯一稳定红色目标 → 到识别目标正上方 100 mm（以高位臂基座 Z 轴和 TCP 为准） → 按新测量小步闭环修正 XY（每步停稳再测，Z 和姿态固定）→ 张爪 → 沿基座 Z 下降 → 低力闭爪 → 垂直抬升并保持夹持。闭爪用宽度/力反馈，不代表已经视觉验证果实在爪中。

复位和观察位通过 `/piper_test/joint_ctrl_single` 发送关节目标，抓取和伺服通过 `/piper_test/pos_cmd` 发送末端目标；使能和夹爪走 ROS 服务。应用层不直接调用 Piper SDK，底层指令顺序由现有 ROS 驱动处理。CAN 监听只用于读取和验证反馈。

## 现场配置

- 机械臂基座 XY 平面需与台面平行；假定夹爪接近方向为 link6 +Z，`[180,0,yaw]` 使其对准基座 -Z。先用只对准模式确认现场方向与视野。
- `work_surface_z_mm` 可留空；若填写实测台面高度，则额外检查台面净空。应用层不设 Link6 或 TCP 工作空间盒；以驱动的执行反馈和到位确认为准。模型 TCP 初值为 110.59 mm，请确认 TCP 和手眼标定适用于这台机械臂。
- 张爪后从伺服对齐时的实际 TCP 位姿沿机械臂基座 Z 轴固定下降 110 mm。RGB-D 深度用于计算悬停位，伺服测得的新深度不改变这段下探距离。
- 高位臂：`can_piper_high` / 1 Mbps，D435i `317622073937`，手眼矩阵 `00_Dual_Arm/calibration/results/T_ee_cam_high.npy`。本流程由高位臂观察和抓取，不需要双臂基座变换。夹爪 yaw 沿用高位臂标定测试的 `-177.856°`。
- 使用本工程的 `piper_high.launch`；ROS 命名空间仍是 `/piper_test`，与旧驱动不能同时运行。该启动文件默认启动一次高位臂复位，使能后用 MOVE_J、10% 速度到六轴 0°，夹爪不动作。启动前确认复位路径净空；仅需启动驱动时加 `startup_reset:=false`。复位成功会打印 `Startup reset complete: high_zero`。退出独立 SDK、idle/run 等其他机械臂执行器。

只启用高位臂接口，再启动对应 ROS 驱动（录制观察位只需 CAN 反馈，不要求启动 ROS 驱动）：

```bash
sudo ip link set dev can_piper_high type can bitrate 1000000
sudo ip link set dev can_piper_high up
source /opt/ros/melodic/setup.bash
source ~/piper_ws/devel/setup.bash --extend
roslaunch ~/tomato_grasp/piper_high.launch
```

```bash
cd ~/tomato_grasp
# 手动把机械臂放在观察位并停稳。只读反馈，--save 才写 config.json。
./run.sh record-observation --save
# 首轮验证：到观察位和目标上方并闭环对准，不动夹爪、不下降。
./run.sh check-config --align-only
./run.sh run --execute --accept-model-tcp --align-only
# 核实垂直路径后，执行完整抓取：
./run.sh check-config && ./run.sh run --execute --accept-model-tcp
```

完整抓取在自动伺服对准且下探、抬升位置检查通过后，立即张爪、下降、闭爪、抬升，不等待按键。`--align-only` 仅用于调试：完成对准后只显示实时画面，按 Q / Esc 或关闭窗口退出；预览不会再次发运动指令。红十字是识别中心，绿圈是计划 TCP，青色叉是反馈推算的 TCP，后两者都投影到目标所在高度。无界面可加 `--no-display`。

抬升停稳后比较张爪和持物阶段各 5 帧夹爪反馈的中值：`effort` 幅值增加至少 0.20 且实测开度比闭爪设定大于 2 mm，判为夹到。低力且开度接近闭爪设定时，回观察位重试一次；连续两次空夹报 `GRASP_EMPTY_AFTER_RETRY`。红色目标不稳定、反馈异常或判定不明确时不重试。已有运动命令时会检查停止反馈；`stop_unconfirmed` 需要现场检查。运行配置、步骤和 CAN 反馈在 `runs/`。

```bash
/usr/bin/python3 -m unittest discover -s tests -v
```

## 只到观察位并查看红色识别

启用高位 CAN、启动上面的 `piper_high.launch`，等待 `Startup reset complete: high_zero`。如果关闭了启动复位，按现场流程使能机械臂。
在图形桌面终端运行：

```bash
cd ~/tomato_grasp
./observe_red.sh --execute
```

脚本使用 `observation_joints_deg` 以 MOVE_J 到观察位，确认停稳后启动 D435i。
窗口左侧为红色 mask、检测框、中心点和三维测量，右侧为对齐后的深度。
按 Q / Esc 或关闭窗口退出。此步骤不动夹爪、不保存截图，不要求台面高度或 TCP 标定。
