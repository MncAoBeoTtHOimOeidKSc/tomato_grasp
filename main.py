#!/usr/bin/env python3
"""Entry point for a single grasp using the existing Piper ROS node."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
from grasp import EmptyGrasp, Grasp, base_point, project_base_point, rotation_distance

ROOT = Path(__file__).resolve().parent
TOOLS = Path.home()/'piper_ws/src/piper_ros/local_tools'
VISION = Path.home()/'robot_project/VLM-to-Robot/visual_servo'
PIPER_DRIVER = TOOLS.parent/'src/piper/scripts'
HIGH_DRIVER = '/piper_test/piper_basic_grasp'


def validate_high_arm(cfg):
    if cfg.get('can_interface') != 'can_piper_high' or cfg['camera'].get('serial') != '317622073937':
        raise ValueError('HIGH_ARM_REQUIRED: can_piper_high / D435i 317622073937')


def validate_config(cfg, align_only=False):
    validate_high_arm(cfg)
    def vector(key, n):
        v = cfg[key]
        if not isinstance(v, list) or len(v) != n or any(type(x) not in (int, float) or not np.isfinite(x) for x in v):
            raise ValueError('INVALID_' + key.upper())
    if cfg['observation_pose_mm_deg'] is None:
        raise ValueError('OBSERVATION_POSE_NOT_RECORDED: run record-observation first')
    vector('observation_pose_mm_deg', 6)
    vector('tcp_offset_ee_mm', 3)
    vector('target_offset_base_mm', 3)
    surface_z = cfg.get('work_surface_z_mm')
    if surface_z is not None and (type(surface_z) not in (int, float) or not -200 <= surface_z <= 500):
        raise ValueError('INVALID_WORK_SURFACE_Z_MM')
    for key, low, high in (
        ('grasp_yaw_deg', -180, 180),
        ('tcp_surface_clearance_mm', 1, 100), ('hover_mm', 70, 300), ('lift_mm', 20, 200),
        ('max_move_mm', 10, 450), ('servo_step_mm', 2, 15), ('servo_gain', .1, 1),
        ('servo_xy_tolerance_mm', 2, 10), ('servo_max_steps', 1, 30), ('servo_timeout_s', 1, 90),
        ('lock_frames', 3, 30), ('verify_frames', 2, 15), ('jitter_mm', 1, 15),
        ('reassociate_mm', 5, 70), ('sample_timeout_s', 1, 20), ('arm_speed_percent', 1, 10),
        ('fine_speed_percent', 1, 10), ('motion_timeout_s', 1, 60), ('gripper_open_mm', 5, 65),
        ('gripper_close_mm', 0, 60), ('gripper_effort', 500, 1000)):
        v = cfg[key]
        integral = key.endswith(('_frames', '_steps', '_percent'))
        if type(v) not in ((int,) if integral else (int, float)) or not low <= v <= high:
            raise ValueError('INVALID_' + key.upper())
    if cfg['gripper_close_mm'] >= cfg['gripper_open_mm']:
        raise ValueError('INVALID_GRIPPER_WIDTHS')
    handeye = np.load(str(Path(cfg['handeye_path']).expanduser()), allow_pickle=False)
    if (handeye.shape != (4, 4) or not np.isfinite(handeye).all() or
            not np.allclose(handeye[3], [0, 0, 0, 1]) or
            not np.allclose(handeye[:3, :3].T @ handeye[:3, :3], np.eye(3), atol=.001) or
            not np.isclose(np.linalg.det(handeye[:3, :3]), 1., atol=.001)):
        raise ValueError('INVALID_HANDEYE')
    return handeye


@contextmanager
def hardware_lock():
    path = Path.home()/'.local/state/vlm-jetson/hardware.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('HARDWARE_BUSY: exit idle/run normally before using tomato_grasp')
        yield


def observation_joints(cfg):
    joints = cfg.get('observation_joints_deg')
    if joints is None:
        raise ValueError('OBSERVATION_JOINTS_NOT_RECORDED: run record-observation --save')
    if not isinstance(joints, list) or len(joints) != 6 or any(
            type(v) not in (int, float) or not np.isfinite(v) for v in joints):
        raise ValueError('INVALID_OBSERVATION_JOINTS_DEG')
    return joints


def preview_red(cfg, camera_factory, draw_location, depth_colormap, cv2, check_state, exiting):
    window = 'Tomato observation | RGB + depth | Q / Esc to exit'
    camera = camera_factory(cfg)
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while not exiting():
            check_state()
            color, depth, result = camera.read()
            if exiting():
                break
            check_state()
            camera_cfg = cfg['camera']
            colored_depth = depth_colormap(depth, camera_cfg['min_depth_m'], camera_cfg['max_depth_m'])
            if result['valid']:
                u, v = result['pixel_center']
                cv2.drawMarker(colored_depth, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.imshow(window, np.hstack((draw_location(color, result), colored_depth)))
            if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            camera.stop()
        finally:
            cv2.destroyAllWindows()


def run_hardware(args, cfg, handeye):
    sys.path[:0] = [str(TOOLS), str(PIPER_DRIVER), str(VISION)]
    import rospy
    from ros_motion_smoke import Monitor, healthy
    from piper_feedback_compat import detect_feedback_layout
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancel.set())

    def check_cancel():
        if cancel.is_set() or rospy.is_shutdown():
            raise RuntimeError('CANCELLED')

    preview = args.mode == 'observe-red'
    motion_mode = preview or args.mode == 'run'
    if motion_mode:
        import os
        if (preview or not args.no_display) and not os.environ.get('DISPLAY'):
            raise RuntimeError('NO_DISPLAY: use a graphical desktop' if preview else
                               'NO_DISPLAY: pass --no-display for headless operation')
        # Import and validate dependencies before moving to the observation pose.
        import cv2
        from locate_color_target import ColorTargetLocator, draw_location, depth_colormap
        from basic_grasp import Sequence, gripper_state
        from fake_harvest_skills import ROSArm
        from std_srvs.srv import SetBool
        rospy.init_node('tomato_grasp', anonymous=True, disable_signals=True)

    logdir = ROOT/'runs'/('{}_{}'.format(args.mode, time.time_ns() if hasattr(time, 'time_ns') else int(time.time()*1e6)))
    logdir.mkdir(parents=True)
    with hardware_lock(), (logdir/'events.jsonl').open('w') as log:
        (logdir/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
        if handeye is not None:
            np.save(str(logdir/'T_ee_cam.npy'), handeye, allow_pickle=False)

        def record(event, **data):
            line = json.dumps(dict(event=event, time=time.time(), **data), allow_nan=False)
            print(line, flush=True)
            log.write(line+'\n'); log.flush()

        can_interface = cfg['can_interface']
        if motion_mode:
            actual_port = rospy.get_param(HIGH_DRIVER+'/can_port', None)
            if actual_port is None:
                raise RuntimeError('PIPER_DRIVER_NOT_CONFIGURED: start roslaunch ~/tomato_grasp/piper_high.launch '
                                   'on the same ROS_MASTER_URI as this script')
            if actual_port != can_interface:
                raise RuntimeError('ROS_CAN_INTERFACE_MISMATCH: expected {}, got {}'.format(can_interface, actual_port))
        offset, _, _ = detect_feedback_layout(can_interface)
        record('arm_feedback_layout', can_interface=can_interface, feedback_offset=offset)
        monitor = Monitor(logdir/'can.jsonl', can_interface=can_interface, feedback_offset=offset)
        arm = None
        commanded = False
        observation_settled = False
        try:
            until = time.monotonic()+4.
            while True:
                check_cancel()
                try:
                    state = monitor.state()
                    break
                except RuntimeError:
                    if time.monotonic() >= until:
                        raise
                    time.sleep(.05)
            healthy(state, require_can=motion_mode)
            if args.mode == 'record-observation':
                # Read only: no ROS driver services, publishers, enable or motion.
                for _ in range(6):
                    check_cancel()
                    time.sleep(.06)
                    new = monitor.state()
                    healthy(new, require_can=False)
                    if (new['stamp'] <= state['stamp'] or new['motion_status'] != 0 or
                        max(abs(a-b) for a,b in zip(new['joints'],state['joints'])) > .12):
                        raise RuntimeError('OBSERVATION_POSE_NOT_STATIONARY')
                    state = new
                pose = pose_mm(state).tolist()
                record('observation_recorded', pose_mm_deg=pose, joints_deg=state['joints'], saved=args.save)
                if args.save:
                    cfg['observation_pose_mm_deg'] = pose
                    cfg['observation_joints_deg'] = state['joints']
                    tmp = args.config.with_suffix('.recording.tmp')
                    tmp.write_text(json.dumps(cfg, indent=2)+'\n')
                    tmp.replace(args.config)
                return

            arm = ROSArm(monitor, record)
            adapter_cfg = dict(gripper_timeout_s=5., gripper_settle_s=.6, gripper_contact_min_effort=.2)
            seq = Sequence(arm, adapter_cfg, {'one_shot_grasp': {'gripper_effort': cfg['gripper_effort']}}, record, cancel)
            arm.prepare(False)
            arm.assert_commands_unblocked()
            for flag in ('gripper_command_in_motion', 'gripper_on_arm_enable'):
                keys = [k for k in rospy.get_param_names() if k.startswith('/piper_test/') and k.endswith('/'+flag)]
                if len(keys)!=1 or rospy.get_param(keys[0]) is not False:
                    raise RuntimeError('DRIVER_REQUIRES_'+flag+'_FALSE')

            if preview:
                joints = observation_joints(cfg)
                speed = cfg['arm_speed_percent']
                timeout = cfg['motion_timeout_s']
                if type(speed) is not int or not 1 <= speed <= 10:
                    raise ValueError('INVALID_ARM_SPEED_PERCENT')
                if type(timeout) not in (int, float) or not 1 <= timeout <= 60:
                    raise ValueError('INVALID_MOTION_TIMEOUT_S')
                until = time.monotonic()+4.
                while arm.jp.get_num_connections() == 0:
                    check_cancel()
                    if time.monotonic() >= until:
                        raise RuntimeError('ROS_JOINT_COMMAND_SUBSCRIBER_MISSING')
                    time.sleep(.05)
                seq.check()
                record('step_start', step='observe', joints_deg=joints, speed_percent=speed)
                status, code, effect = arm.move_joints(joints, speed, int(timeout*1000), cancel)
                commanded = effect.get('command_sent') is True
                record('step_result', step='observe', status=status, error_code=code, effect=effect)
                if status != 'SUCCEEDED' or effect.get('stopped') is not True:
                    raise RuntimeError('OBSERVATION_MOVE_FAILED: {}: {}'.format(code, effect.get('original_error', '')))

                def check_observation():
                    current = seq.check()
                    if current['motion_status'] != 0 or max(abs(a-b) for a,b in zip(current['joints'],joints)) > .5:
                        raise RuntimeError('ARM_NOT_AT_STATIONARY_OBSERVATION_POSE')

                check_observation()
                observation_settled = True
                record('preview_start', camera_serial=cfg['camera']['serial'])
                try:
                    preview_red(cfg, ColorTargetLocator, draw_location, depth_colormap, cv2,
                                check_observation, lambda: cancel.is_set() or rospy.is_shutdown())
                except RuntimeError:
                    if not (cancel.is_set() or rospy.is_shutdown()):
                        raise
                record('preview_closed')
                return

            class Arm:
                def pose(self):
                    return pose_mm(seq.check())

                def move(self, name, target, fine):
                    nonlocal commanded
                    check_cancel()
                    record('step_start', step=name, pose_mm_deg=target.tolist())
                    status, code, effect = arm.move_end_pose(**dict(zip(
                        ('x_mm','y_mm','z_mm','rx_deg','ry_deg','rz_deg'), map(float, target))),
                        speed_percent=cfg['fine_speed_percent' if fine else 'arm_speed_percent'],
                        timeout_ms=int(cfg['motion_timeout_s']*1000), cancel=cancel)
                    commanded |= effect.get('command_sent') is True
                    record('step_result', step=name, status=status, error_code=code, effect=effect)
                    if status != 'SUCCEEDED' or effect.get('stopped') is not True:
                        raise RuntimeError('{}: {}: {}'.format(name, code, effect.get('original_error', '')))

                def grip(self, name, width, contact=False):
                    nonlocal commanded
                    check_cancel()
                    commanded = True
                    seq.grip(name, width, contact=contact)

                def grip_feedback(self):
                    samples = []
                    stamp = gripper_state(monitor)['stamp']
                    deadline = time.monotonic() + 1.
                    while time.monotonic() < deadline:
                        seq.check()
                        state = gripper_state(monitor)
                        if not state['enabled']:
                            raise RuntimeError('GRIPPER_NOT_ENABLED')
                        if state['stamp'] > stamp:
                            stamp = state['stamp']
                            samples.append(state)
                            if len(samples) == 5:
                                return {key: float(np.median([s[key] for s in samples]))
                                        for key in ('width_mm', 'effort')}
                        time.sleep(.05)
                    raise RuntimeError('GRIPPER_FEEDBACK_STALE')

            class Camera:
                def __init__(self):
                    self.locator = ColorTargetLocator(cfg)
                    self.last_stamp = None
                    record('camera_started', serial=cfg['camera']['serial'])

                def measure(self, _):
                    before = seq.check()
                    start = time.monotonic()
                    bgr, _, result = self.locator.read()
                    after = seq.check()
                    if not args.no_display:
                        cv2.imshow('tomato_grasp', draw_location(bgr, result))
                        if (cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')) or
                                cv2.getWindowProperty('tomato_grasp', cv2.WND_PROP_VISIBLE) < 1):
                            cancel.set(); check_cancel()
                    stamp = result['camera_timestamp_s']
                    if self.last_stamp is not None and stamp <= self.last_stamp:
                        return None
                    self.last_stamp = stamp
                    p, q = pose_mm(before), pose_mm(after)
                    if (time.monotonic()-start > .15 or np.linalg.norm(p[:3]-q[:3]) > 1. or
                        rotation_distance(p[3:],q[3:]) > .5 or after['motion_status'] != 0 or
                        not result['valid'] or result['candidate_count'] != 1):
                        return None
                    return base_point(q, handeye, np.asarray(result['camera_xyz_m']))

                def draw_tcp_overlay(self, frame, result, before, after, elapsed):
                    text = 'TCP overlay unavailable: need a stable arm and one RGB-D target'
                    if before is not None and after is not None:
                        p, q = pose_mm(before), pose_mm(after)
                        if (elapsed <= .15 and after['motion_status'] == 0 and
                                np.linalg.norm(p[:3] - q[:3]) <= 1. and
                                rotation_distance(p[3:], q[3:]) <= .5 and
                                result['valid'] and result['candidate_count'] == 1):
                            target = base_point(q, handeye, np.asarray(result['camera_xyz_m']))
                            goal = target + cfg['target_offset_base_mm']
                            tcp = grasp.tcp_for_pose(q)
                            delta = goal[:2] - tcp[:2]
                            # Compare XY on the same plane, so camera perspective does
                            # not make equal XY look different at hover height.
                            planned_uv = project_base_point(np.r_[goal[:2], target[2]], q,
                                                            handeye, self.locator.intrinsics)
                            feedback_uv = project_base_point(np.r_[tcp[:2], target[2]], q,
                                                             handeye, self.locator.intrinsics)
                            if planned_uv is not None:
                                cv2.circle(frame, planned_uv, 12, (0, 255, 0), 2, cv2.LINE_AA)
                            if feedback_uv is not None:
                                cv2.drawMarker(frame, feedback_uv, (255, 255, 0),
                                               cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
                            text = 'XY error {:+.1f}, {:+.1f} mm | {:.1f} mm'.format(
                                delta[0], delta[1], np.linalg.norm(delta))
                            if planned_uv is None or feedback_uv is None:
                                text += ' | marker off-screen'
                    cv2.putText(frame, text, (12, frame.shape[0] - 37), cv2.FONT_HERSHEY_SIMPLEX,
                                .47, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, 'Green: planned TCP | Cyan: feedback TCP (target plane)',
                                (12, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                .47, (255, 255, 255), 1, cv2.LINE_AA)

                def preview_state(self):
                    try:
                        return monitor.state()
                    except RuntimeError:
                        return None

                def show_aligned(self):
                    window = 'tomato_grasp'
                    message = 'Alignment complete | Read-only preview'
                    while not (cancel.is_set() or rospy.is_shutdown()):
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                        before = self.preview_state()
                        start = time.monotonic()
                        bgr, _, result = self.locator.read()
                        after = self.preview_state()
                        frame = draw_location(bgr, result)
                        self.draw_tcp_overlay(frame, result, before, after, time.monotonic() - start)
                        cv2.putText(frame, message, (12, 101), cv2.FONT_HERSHEY_SIMPLEX,
                                    .6, (0, 255, 0), 2)
                        cv2.putText(frame, 'Q / Esc to exit', (12, 125), cv2.FONT_HERSHEY_SIMPLEX,
                                    .55, (255, 255, 255), 2)
                        cv2.imshow(window, frame)
                        key = cv2.waitKey(1) & 0xff
                        if key in (27, ord('q'), ord('Q')):
                            break

                def close(self):
                    self.locator.stop()
                    if not args.no_display:
                        cv2.destroyAllWindows()

            grasp = Grasp(cfg, Arm(), Camera, record, check_cancel)
            grasp.validate_pose(cfg['observation_pose_mm_deg'])
            # Start only with an already enabled arm in CAN mode. Do not change modes or enable here.
            seq.check(); gripper_state(monitor)
            def record_completion(hold_feedback=None):
                record('complete', alignment_only=args.align_only, physical_grasp_success_verified=False,
                       gripper_contact_detected=hold_feedback is not None,
                       gripper=hold_feedback if hold_feedback is not None else gripper_state(monitor))

            def keep_alignment_preview(camera, hover_z):
                record_completion()
                record('alignment_preview_start')

                try:
                    camera.show_aligned()
                except (cv2.error, RuntimeError, OSError) as exc:
                    record('alignment_preview_error', error=str(exc))
                finally:
                    record('alignment_preview_closed')

            keep_preview = args.align_only and not args.no_display
            feedback = grasp.run(align_only=args.align_only,
                                 on_aligned=keep_alignment_preview if keep_preview else None)
            if not keep_preview:
                record_completion(feedback)
        except EmptyGrasp as exc:
            record('failed', error=str(exc), command_sent=commanded, stopped=True)
            raise
        except BaseException as exc:
            record('failed', error=str(exc), command_sent=commanded)
            if commanded and arm is not None and not observation_settled:
                try:
                    response = rospy.ServiceProxy('/piper_test/block_arm', SetBool)(True)
                    if not response.success:
                        raise RuntimeError(response.message)
                except Exception as err:
                    record('block_error', error=str(err))
                try:
                    state = arm.stationary()
                    first=gripper_state(monitor)
                    time.sleep(.4)
                    last=gripper_state(monitor)
                    if state['motion_status'] != 0 or last['stamp'] <= first['stamp'] or abs(last['width_mm']-first['width_mm']) > .2:
                        raise RuntimeError('ARM_OR_GRIPPER_STOP_UNCONFIRMED')
                    record('stationary_feedback', state=state, gripper=last)
                except Exception as err:
                    record('stop_unconfirmed', error=str(err))
            raise
        finally:
            monitor.close()


def pose_mm(state):
    return np.asarray(state['pose']) * [1000,1000,1000,1,1,1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    modes = parser.add_subparsers(dest='mode')
    check = modes.add_parser('check-config', help='validate geometry config offline')
    check.add_argument('--align-only', action='store_true', help='compatibility option; surface height is optional in either mode')
    teach = modes.add_parser('record-observation', help='read stationary pose without motion')
    teach.add_argument('--save', action='store_true')
    preview = modes.add_parser('observe-red', help='move to observation joints and show red RGB-D preview')
    preview.add_argument('--execute', action='store_true')
    run = modes.add_parser('run', help='align and automatically execute one pickup')
    run.add_argument('--execute', action='store_true')
    run.add_argument('--accept-model-tcp', action='store_true')
    run.add_argument('--no-display', action='store_true')
    run.add_argument('--align-only', action='store_true', help='stop after alignment without moving the gripper')
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return
    if args.mode == 'run' and not (args.execute and args.accept_model_tcp):
        parser.error('run requires --execute and --accept-model-tcp')
    if args.mode == 'observe-red' and not args.execute:
        parser.error('observe-red requires --execute')
    cfg = json.loads(args.config.read_text())
    validate_high_arm(cfg)
    if args.mode == 'observe-red':
        observation_joints(cfg)
    handeye = validate_config(cfg, align_only=args.align_only) if args.mode in ('run', 'check-config') else None
    if args.mode == 'check-config':
        Grasp(cfg, None, None, None, None).validate_pose(cfg['observation_pose_mm_deg'])
        print('CONFIG_OK')
    else:
        run_hardware(args, cfg, handeye)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        sys.exit(str(exc))
