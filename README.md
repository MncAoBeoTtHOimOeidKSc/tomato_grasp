# tomato_grasp

单次垂直抓取。复用 visual_servo 的 红色 HSV 定位（相机为高位臂的 D435i）和手眼矩阵，复用 piper_ws 的 ROSArm 状态确认与夹爪反馈，不启动 SAM/YOLO 或新驱动。

流程：到观察位并确认到位 → 打开 RGB-D、锁定唯一稳定红色目标 → 到识别目标正上方 100 mm（以高位臂基座 Z 轴和 TCP 为准） → 按新测量小步闭环修正 XY（每步停稳再测，Z 和姿态固定）→ 张爪 → 沿基座 Z 下降 → 低力闭爪 → 垂直抬升并保持夹持。闭爪用宽度/力反馈，不代表已经视觉验证果实在爪中。

复位和观察位通过 `/piper_test/joint_ctrl_single` 发送关节目标，抓取和伺服通过 `/piper_test/pos_cmd` 发送末端目标；使能和夹爪走 ROS 服务。应用层不直接调用 Piper SDK，底层指令顺序由现有 ROS 驱动处理。CAN 监听只用于读取和验证反馈。

## 现场配置

- 机械臂基座 XY 平面需与台面平行；假定夹爪接近方向为 link6 +Z，`[180,0,yaw]` 使其对准基座 -Z。先用只对准模式确认现场方向与视野。
- 完整抓取时，`work_surface_z_mm` 填台面在基座坐标中的实测高度；`--align-only` 可留空。还要确认 TCP、手眼标定适用于这台手臂，模型 TCP 初值为 110.59 mm。TCP 工作空间 Z 下限设为 `null`，允许基座坐标为负的 TCP 高度。其他工作空间边界及台面净空检查仍生效；这些检查不能证明轨迹无碰撞。
- 夹取深度用对准后的 RGB-D 表面位置加 `target_offset_base_mm`；默认补偿为零。物体表面和实际夹取点不重合时按现场测量调整，不沿用旧姿态的经验补偿。
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
# 填好 work_surface_z_mm 并核实垂直路径后，执行完整抓取：
./run.sh check-config
./run.sh run --execute --accept-model-tcp
```

带界面运行 `--align-only` 时，对准成功后继续显示实时识别画面，按 Q / Esc 或关闭窗口退出。预览默认不发运动指令。移动目标并将手撤出工作区后，按 R 重新识别并对准；它保持首次悬停的 Z 和夹爪姿态，按稳定测量结果最多发送一条 XY 修正指令，再复测误差；若未达标，可重新按 R，不会自动补发。单次修正距离不得超过原先的累计步距上限。目标无法稳定锁定或超出软件边界时可留在预览，检查后再次按 R。已下发的运动若报故障，本轮只保留预览，不重试或自动解除驱动阻断。预览中的红十字是识别中心，绿圈是实时估算的计划 TCP，青色叉是机械臂反馈推算的 TCP；后两者都投影到目标所在高度，便于比较 XY。残差以毫米显示，仍依赖相机、手眼和 TCP 标定。无 GUI 可加 `--no-display`，对准完成后直接退出。有多个有效红色目标、目标出画或深度丢失时不会盲目下探。失败后不自动重试复位；已有运动命令时阻断后续 ROS 应用命令并检查停止反馈，`stop_unconfirmed` 需要现场检查。运行配置、步骤和 CAN 反馈在 `runs/`。

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
