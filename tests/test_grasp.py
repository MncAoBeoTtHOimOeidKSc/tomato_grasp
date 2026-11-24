import json
from pathlib import Path
import unittest

import numpy as np
from grasp import Grasp, base_point, rotation, rotation_distance
from main import validate_config

ROOT = Path(__file__).resolve().parents[1]


class FakeArm:
    def __init__(self):
        self.current = np.array([250., 0., 400., 175., 30., 175.])
        self.calls = []

    def pose(self):
        return self.current.copy()

    def move(self, name, pose, fine):
        self.calls.append((name, pose.copy(), fine))
        self.current = pose.copy()

    def grip(self, name, width, contact=False):
        self.calls.append((name, width, contact))


class FakeCamera:
    def __init__(self, arm, point):
        self.arm, self.point = arm, point
        self.closed = False
        self.before_hover = np.array([350., 20., 45.])

    def measure(self, arm):
        return self.before_hover if self.arm.calls[-1][0] == 'observe' else self.point

    def close(self):
        self.closed = True


class GraspTests(unittest.TestCase):
    def setup_run(self, point=(361, 25, 44)):
        cfg = json.loads((ROOT/'config.json').read_text())
        cfg.update(observation_pose_mm_deg=[250., 0., 400., 175., 30., 175.], work_surface_z_mm=10.)
        arm = FakeArm()
        counter = iter(np.arange(0, 500, .1))
        camera = FakeCamera(arm, None if point is None else np.array(point, dtype=float))
        def start_camera():
            self.assertEqual([c[0] for c in arm.calls], ['observe'])
            return camera
        g = Grasp(cfg, arm, start_camera, lambda *a, **kw: None, lambda: None, lambda: next(counter))
        return cfg, arm, camera, g

    def test_feedback_corrects_xy_then_descends_and_lifts_at_same_xy(self):
        cfg, arm, camera, g = self.setup_run()
        g.run()
        names = [c[0] for c in arm.calls]
        self.assertEqual(names[:2], ['observe', 'hover'])
        self.assertIn('servo_xy', names)
        self.assertEqual(names[-4:], ['open', 'descend', 'close', 'lift'])
        hover = g.tcp_for_pose(arm.calls[1][1])
        np.testing.assert_allclose(hover, camera.before_hover + [0, 0, 100], atol=1e-8)
        prev = hover
        for name, pose, fine in arm.calls[2:-4]:
            self.assertEqual(name, 'servo_xy')
            current = g.tcp_for_pose(pose)
            self.assertAlmostEqual(current[2], hover[2])
            self.assertLessEqual(np.linalg.norm(current-prev), cfg['servo_step_mm']+1e-6)
            prev = current
        descend = g.tcp_for_pose(arm.calls[-3][1])
        lift = g.tcp_for_pose(arm.calls[-1][1])
        np.testing.assert_allclose(descend[:2], lift[:2])
        self.assertLessEqual(np.linalg.norm(descend[:2]-camera.point[:2]), cfg['servo_xy_tolerance_mm'])
        self.assertAlmostEqual(lift[2]-descend[2], cfg['lift_mm'])
        self.assertTrue(arm.calls[-2][2])
        self.assertTrue(camera.closed)

    def test_hover_outside_workspace_is_not_dispatched(self):
        cfg, arm, camera, g = self.setup_run()
        camera.before_hover[2] = cfg["link6_workspace_mm"][2][1] - cfg["hover_mm"] + 1.
        with self.assertRaisesRegex(RuntimeError, "LINK6_WORKSPACE_MM|TCP_WORKSPACE_MM"):
            g.run(align_only=True)
        self.assertEqual([c[0] for c in arm.calls], ['observe'])
        self.assertTrue(camera.closed)

    def test_negative_tcp_hover_is_allowed_but_surface_clearance_still_applies(self):
        cfg, arm, camera, g = self.setup_run([478.28, 18.07, -146.25])
        camera.before_hover[:] = camera.point
        cfg['work_surface_z_mm'] = None
        g.run(align_only=True)
        hover_pose = arm.calls[1][1]
        np.testing.assert_allclose(g.tcp_for_pose(hover_pose), [478.28, 18.07, -46.25])
        cfg['work_surface_z_mm'] = -50.
        with self.assertRaisesRegex(RuntimeError, 'TCP_BELOW_CONFIGURED_SURFACE_CLEARANCE'):
            g.validate_pose(hover_pose)

    def test_align_only_without_surface_height_leaves_gripper_alone(self):
        cfg, arm, camera, g = self.setup_run()
        cfg['work_surface_z_mm'] = None
        g.run(align_only=True)
        self.assertTrue(all(c[0] in ('observe','hover','servo_xy') for c in arm.calls))
        self.assertTrue(camera.closed)

    def test_realign_relocks_moved_target_without_changing_height_or_gripper(self):
        cfg, arm, camera, g = self.setup_run()

        def after_alignment(active_camera, hover_z):
            self.assertIs(active_camera, camera)
            self.assertFalse(camera.closed)
            start = len(arm.calls)
            camera.point = np.array([416., 40., 55.])
            tcp, target = g.realign_xy(hover_z)
            self.assertLessEqual(np.linalg.norm(tcp[:2] - target[:2]), cfg['servo_xy_tolerance_mm'])
            self.assertEqual(len(arm.calls), start + 1)
            for name, pose, fine in arm.calls[start:]:
                self.assertEqual(name, 'servo_xy')
                self.assertTrue(fine)
                self.assertAlmostEqual(g.tcp_for_pose(pose)[2], hover_z)
                np.testing.assert_allclose(pose[3:], g.rpy)

        g.run(align_only=True, on_aligned=after_alignment)
        self.assertTrue(camera.closed)

    def test_realign_unstable_or_out_of_bounds_target_does_not_dispatch(self):
        _, arm, camera, g = self.setup_run()

        def after_alignment(active_camera, hover_z):
            start = len(arm.calls)
            for point, error in ((None, 'TARGET_LOST_OR_UNSTABLE'),
                                 (np.array([680., 25., 44.]), 'LINK6_WORKSPACE_MM'),
                                 (np.array([500., 25., 44.]), 'REALIGN_DISTANCE_EXCEEDED')):
                active_camera.point = point
                with self.assertRaisesRegex(RuntimeError, error):
                    g.realign_xy(hover_z)
                self.assertEqual(len(arm.calls), start)
                self.assertFalse(active_camera.closed)

        g.run(align_only=True, on_aligned=after_alignment)

    def test_realign_residual_does_not_send_another_command(self):
        _, arm, camera, g = self.setup_run()

        def after_alignment(active_camera, hover_z):
            camera.point = np.array([416., 40., 55.])
            start = len(arm.calls)
            move = arm.move

            def move_then_shift_target(name, pose, fine):
                move(name, pose, fine)
                camera.point = camera.point + [10., 0., 0.]

            arm.move = move_then_shift_target
            with self.assertRaisesRegex(RuntimeError, 'REALIGN_RESIDUAL_OUT_OF_TOLERANCE'):
                g.realign_xy(hover_z)
            self.assertEqual(len(arm.calls), start + 1)

        g.run(align_only=True, on_aligned=after_alignment)

    def test_full_grasp_without_surface_height_cannot_move(self):
        cfg, arm, _, g = self.setup_run()
        cfg['work_surface_z_mm'] = None
        with self.assertRaisesRegex(ValueError, 'WORK_SURFACE_HEIGHT_REQUIRED_FOR_GRASP'):
            g.run()
        self.assertEqual(arm.calls, [])

    def test_lost_or_different_target_stops_before_opening(self):
        for point in (None, [410., 20., 45.]):
            _, arm, camera, g = self.setup_run(point)
            with self.assertRaisesRegex(RuntimeError, 'TARGET_LOST_OR_UNSTABLE'):
                g.run()
            self.assertEqual([c[0] for c in arm.calls], ['observe', 'hover'])
            self.assertTrue(camera.closed)

    def test_low_pick_is_rejected_without_opening(self):
        _, arm, camera, g = self.setup_run([350., 20., 15.])
        with self.assertRaisesRegex(RuntimeError, 'SURFACE_CLEARANCE'):
            g.run()
        self.assertEqual([c[0] for c in arm.calls], ['observe', 'hover'])
        self.assertTrue(camera.closed)

    def test_failed_move_does_not_grip_or_lift(self):
        _, arm, _, g = self.setup_run()
        dispatch = arm.move
        def move(name, pose, fine):
            if name == 'descend':
                raise RuntimeError('ARM_FAULT')
            dispatch(name, pose, fine)
        arm.move = move
        with self.assertRaisesRegex(RuntimeError, 'ARM_FAULT'):
            g.run()
        self.assertEqual(arm.calls[-1][0], 'open')

    def test_vertical_axis_and_transform_units(self):
        np.testing.assert_allclose(rotation([180,0,43])[:,2], [0,0,-1], atol=1e-10)
        self.assertLess(rotation_distance([-180,0,0], [180,0,0]), 1e-6)
        handeye = np.eye(4)
        handeye[:3,3] = [.01,0,.02]
        point = base_point([100,0,400,180,0,0], handeye, [0,0,.2])
        np.testing.assert_allclose(point, [110,0,180], atol=1e-10)

    def test_default_unrecorded_configuration_cannot_run(self):
        cfg = json.loads((ROOT/'config.json').read_text())
        cfg['observation_pose_mm_deg'] = None
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_POSE_NOT_RECORDED'):
            validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
