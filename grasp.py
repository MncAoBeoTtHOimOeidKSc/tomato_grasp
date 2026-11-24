"""Hardware-independent geometry and stop-and-look grasp sequence. mm / deg."""
from collections import deque
import time

import numpy as np


def rotation(rpy):
    x, y, z = np.deg2rad(rpy)
    a, b, c = np.cos([x, y, z])
    d, e, f = np.sin([x, y, z])
    return np.array([[c*b, c*e*d-f*a, c*e*a+f*d],
                     [f*b, f*e*d+c*a, f*e*a-c*d], [-e, b*d, b*a]])


def rotation_distance(a, b):
    return np.rad2deg(np.arccos(np.clip((np.trace(rotation(a).T @ rotation(b))-1)/2, -1, 1)))


def base_point(pose, handeye, camera_xyz_m):
    point_ee_mm = (handeye[:3, :3] @ camera_xyz_m + handeye[:3, 3]) * 1000
    return np.asarray(pose[:3]) + rotation(pose[3:]) @ point_ee_mm


def project_base_point(point_mm, pose, handeye, intrinsics):
    """Project a base-frame point using the locator's color pinhole model."""
    point_ee_m = rotation(pose[3:]).T @ (np.asarray(point_mm) - pose[:3]) / 1000.
    x, y, z = handeye[:3, :3].T @ (point_ee_m - handeye[:3, 3])
    if not np.isfinite([x, y, z]).all() or z <= 0:
        return None
    u = intrinsics['fx'] * x / z + intrinsics['cx']
    v = intrinsics['fy'] * y / z + intrinsics['cy']
    if not (0 <= u < intrinsics['width'] - .5 and 0 <= v < intrinsics['height'] - .5):
        return None
    return int(round(u)), int(round(v))


class Grasp:
    def __init__(self, cfg, arm, camera_factory, record, check_cancel, clock=time.monotonic):
        self.cfg, self.arm, self.camera_factory = cfg, arm, camera_factory
        self.record, self.check_cancel, self.clock = record, check_cancel, clock
        self.rpy = np.array([180., 0., cfg['grasp_yaw_deg']])
        self.offset = np.asarray(cfg['tcp_offset_ee_mm'])
        self.camera = None

    def pose_for_tcp(self, tcp):
        return np.r_[tcp - rotation(self.rpy) @ self.offset, self.rpy]

    def tcp_for_pose(self, pose):
        return np.asarray(pose[:3]) + rotation(pose[3:]) @ self.offset

    def require_box(self, xyz, key):
        if not np.isfinite(xyz).all() or any(
                (lo is not None and v < lo) or v > hi for v, (lo, hi) in zip(xyz, self.cfg[key])):
            raise RuntimeError(key.upper() + ': ' + str(list(xyz)))

    def validate_pose(self, pose):
        self.require_box(pose[:3], 'link6_workspace_mm')
        tcp = self.tcp_for_pose(pose)
        self.require_box(tcp, 'tcp_workspace_mm')
        surface_z = self.cfg.get('work_surface_z_mm')
        if surface_z is not None and tcp[2] < surface_z + self.cfg['tcp_surface_clearance_mm']:
            raise RuntimeError('TCP_BELOW_CONFIGURED_SURFACE_CLEARANCE')

    def move(self, name, pose, fine=False, vertical=False):
        self.check_cancel()
        self.validate_pose(pose)
        if np.linalg.norm(np.asarray(pose[:3]) - self.arm.pose()[:3]) > self.cfg['max_move_mm']:
            raise RuntimeError('MOVE_DISTANCE_EXCEEDED')
        self.arm.move(name, pose, fine)
        actual = self.arm.pose()
        if vertical and rotation_distance(actual[3:], self.rpy) > 2.:
            raise RuntimeError('VERTICAL_ORIENTATION_NOT_CONFIRMED')
        return actual

    def stable_target(self, count, deadline, anchor=None):
        history = deque(maxlen=count)
        while self.clock() < deadline:
            self.check_cancel()
            point = self.camera.measure(self.arm)
            if point is None or not np.isfinite(point).all() or (anchor is not None and
                    np.linalg.norm(point - anchor) > self.cfg['reassociate_mm']):
                history.clear()
                continue
            history.append(point)
            if len(history) == count:
                center = np.median(history, axis=0)
                if max(np.linalg.norm(p - center) for p in history) <= self.cfg['jitter_mm']:
                    return center
        raise RuntimeError('TARGET_LOST_OR_UNSTABLE')

    def aligned_height_tcp(self, hover_z):
        actual = self.arm.pose()
        if rotation_distance(actual[3:], self.rpy) > 2.:
            raise RuntimeError('SERVO_ORIENTATION_DRIFT')
        tcp = self.tcp_for_pose(actual)
        if abs(tcp[2] - hover_z) > 4.:
            raise RuntimeError('SERVO_HEIGHT_DRIFT')
        return tcp

    def align_xy(self, anchor, hover_z):
        cfg = self.cfg
        bias = np.asarray(cfg['target_offset_base_mm'])
        deadline = self.clock() + cfg['servo_timeout_s']
        steps = 0
        while self.clock() < deadline:
            sample_deadline = min(deadline, self.clock() + cfg['sample_timeout_s'])
            target = self.stable_target(cfg['verify_frames'], sample_deadline, anchor) + bias
            tcp = self.aligned_height_tcp(hover_z)
            error = target[:2] - tcp[:2]
            distance = float(np.linalg.norm(error))
            self.record('servo_measurement', error_xy_mm=error.tolist(), target_mm=target.tolist())
            if distance <= cfg['servo_xy_tolerance_mm']:
                break
            if steps >= cfg['servo_max_steps']:
                raise RuntimeError('SERVO_STEP_LIMIT')
            step = error * min(cfg['servo_gain'], cfg['servo_step_mm'] / distance)
            command = np.r_[tcp[:2] + step, hover_z]
            self.move('servo_xy', self.pose_for_tcp(command), fine=True, vertical=True)
            steps += 1
        else:
            raise RuntimeError('SERVO_TIMEOUT')
        return tcp, target

    def realign_xy(self, hover_z):
        cfg = self.cfg
        anchor = self.stable_target(cfg['lock_frames'], self.clock() + cfg['sample_timeout_s'])
        bias = np.asarray(cfg['target_offset_base_mm'])
        target = anchor + bias
        tcp = self.aligned_height_tcp(hover_z)
        error = target[:2] - tcp[:2]
        distance = float(np.linalg.norm(error))
        self.record('servo_measurement', error_xy_mm=error.tolist(), target_mm=target.tolist())
        if distance <= cfg['servo_xy_tolerance_mm']:
            return tcp, target
        goal = self.pose_for_tcp(np.r_[target[:2], hover_z])
        self.validate_pose(goal)
        # Keep the old maximum total correction while sending a single setpoint.
        if distance > cfg['servo_step_mm'] * cfg['servo_max_steps']:
            raise RuntimeError('REALIGN_DISTANCE_EXCEEDED: {:.1f} mm'.format(distance))
        self.move('servo_xy', goal, fine=True, vertical=True)

        target = self.stable_target(cfg['verify_frames'], self.clock() + cfg['sample_timeout_s'], anchor) + bias
        tcp = self.aligned_height_tcp(hover_z)
        error = target[:2] - tcp[:2]
        self.record('servo_measurement', error_xy_mm=error.tolist(), target_mm=target.tolist())
        residual = float(np.linalg.norm(error))
        if residual > cfg['servo_xy_tolerance_mm']:
            raise RuntimeError('REALIGN_RESIDUAL_OUT_OF_TOLERANCE: {:.1f} mm'.format(residual))
        return tcp, target

    def run(self, align_only=False, on_aligned=None):
        cfg = self.cfg
        if not align_only and cfg.get('work_surface_z_mm') is None:
            raise ValueError('WORK_SURFACE_HEIGHT_REQUIRED_FOR_GRASP')
        self.move('observe', np.asarray(cfg['observation_pose_mm_deg']))
        self.check_cancel()
        self.camera = self.camera_factory()
        try:
            bias = np.asarray(cfg['target_offset_base_mm'])
            anchor = self.stable_target(cfg['lock_frames'], self.clock() + cfg['sample_timeout_s'])
            # The coarse waypoint is directly above the detected surface in base coordinates.
            hover = anchor + [0, 0, cfg['hover_mm']]
            # Full grasps also validate the estimated contact point before approaching.
            if not align_only:
                self.validate_pose(self.pose_for_tcp(anchor + bias))
            self.record('hover_planned', target_base_mm=anchor.tolist(), tcp_base_mm=hover.tolist())
            self.move('hover', self.pose_for_tcp(hover), vertical=True)
            tcp, target = self.align_xy(anchor, hover[2])
            if align_only:
                if on_aligned is not None:
                    on_aligned(self.camera, hover[2])
                return
            # Freeze aligned XY and depth for the final vertical-only move.
            descent = np.r_[tcp[:2], target[2]]
            lift = descent + [0, 0, cfg['lift_mm']]
            if not cfg['hover_mm']/2 <= tcp[2] - descent[2] <= cfg['hover_mm'] + cfg['reassociate_mm']:
                raise RuntimeError('DESCENT_DISTANCE_OUT_OF_RANGE')
            for point in (descent, lift):
                self.validate_pose(self.pose_for_tcp(point))
            self.check_cancel()
            self.arm.grip('open', cfg['gripper_open_mm'])
            self.move('descend', self.pose_for_tcp(descent), fine=True, vertical=True)
            self.check_cancel()
            self.arm.grip('close', cfg['gripper_close_mm'], contact=True)
            self.move('lift', self.pose_for_tcp(lift), fine=True, vertical=True)
        finally:
            self.camera.close()
