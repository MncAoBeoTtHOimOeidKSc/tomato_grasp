import json
from pathlib import Path
import unittest

import numpy as np
from grasp import EmptyGrasp, Grasp, base_point, rotation, rotation_distance
from main import validate_config

ROOT = Path(__file__).resolve().parents[1]


class FakeArm:
    def __init__(self):
        self.current = np.array([250., 0., 400., 175., 30., 175.])
        self.calls = []
        self.hold_feedback = iter([dict(width_mm=22.05, effort=-.521)])

    def pose(self):
        return self.current.copy()

    def move(self, name, pose, fine):
        self.calls.append((name, pose.copy(), fine))
        self.current = pose.copy()

    def grip(self, name, width, contact=False):
        self.calls.append((name, width, contact))

    def grip_feedback(self):
        if self.calls[-1][0] == 'open':
            return dict(width_mm=64.19, effort=.069)
        if self.calls[-1][0] == 'lift':
            return next(self.hold_feedback)
        raise AssertionError('Unexpected feedback timing')


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
            self.assertEqual(arm.calls[-1][0], 'observe')
            camera.closed = False
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
        self.assertAlmostEqual(prev[2]-descend[2], 110.)
        lift = g.tcp_for_pose(arm.calls[-1][1])
        np.testing.assert_allclose(descend[:2], lift[:2])
        self.assertLessEqual(np.linalg.norm(descend[:2]-camera.point[:2]), cfg['servo_xy_tolerance_mm'])
        self.assertAlmostEqual(lift[2]-descend[2], cfg['lift_mm'])
        self.assertTrue(arm.calls[-2][2])
        self.assertTrue(camera.closed)

    def test_empty_gripper_reobserves_once_then_accepts_contact(self):
        _, arm, camera, g = self.setup_run()
        arm.hold_feedback = iter([dict(width_mm=5.11, effort=-.132),
                                  dict(width_mm=22.05, effort=-.521)])
        self.assertEqual(g.run()['width_mm'], 22.05)
        names = [c[0] for c in arm.calls]
        self.assertEqual(names.count('observe'), 2)
        self.assertEqual(names.count('lift'), 2)
        self.assertEqual(names[names.index('lift') + 1], 'observe')
        self.assertTrue(camera.closed)

    def test_empty_retry_limit(self):
        _, arm, _, g = self.setup_run()
        arm.hold_feedback = iter([dict(width_mm=5.11, effort=-.132)] * 2)
        with self.assertRaisesRegex(EmptyGrasp, 'GRASP_EMPTY_AFTER_RETRY'):
            g.run()
        self.assertEqual([c[0] for c in arm.calls].count('observe'), 2)

    def test_inconclusive_close_does_not_retry(self):
        _, arm, _, g = self.setup_run()
        arm.hold_feedback = iter([dict(width_mm=22.05, effort=-.132)])
        with self.assertRaisesRegex(RuntimeError, 'GRIPPER_CONTACT_UNCERTAIN'):
            g.run()
        self.assertEqual([c[0] for c in arm.calls].count('observe'), 1)

    def test_stale_gripper_feedback_does_not_retry(self):
        _, arm, _, g = self.setup_run()
        read = arm.grip_feedback

        def fail_after_lift():
            if arm.calls[-1][0] == 'lift':
                raise RuntimeError('GRIPPER_FEEDBACK_STALE')
            return read()

        arm.grip_feedback = fail_after_lift
        with self.assertRaisesRegex(RuntimeError, 'GRIPPER_FEEDBACK_STALE'):
            g.run()
        self.assertEqual([c[0] for c in arm.calls].count('observe'), 1)

    def test_negative_link6_descent_is_forwarded_to_driver(self):
        target = [486.53, -19.78, -136.37]
        cfg, arm, camera, g = self.setup_run(target)
        cfg['work_surface_z_mm'] = None
        camera.before_hover[:] = camera.point
        g.run()
        self.assertEqual([call[0] for call in arm.calls][-4:], ['open', 'descend', 'close', 'lift'])
        self.assertAlmostEqual(arm.calls[-3][1][2], -35.78)

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

    def test_failed_servo_never_opens_gripper(self):
        cfg, arm, camera, g = self.setup_run()
        cfg['servo_max_steps'] = 1
        move = arm.move

        def miss_servo(name, pose, fine):
            if name == 'servo_xy':
                arm.calls.append((name, pose.copy(), fine))
            else:
                move(name, pose, fine)

        arm.move = miss_servo
        with self.assertRaisesRegex(RuntimeError, 'SERVO_STEP_LIMIT'):
            g.run()
        self.assertEqual([c[0] for c in arm.calls], ['observe', 'hover', 'servo_xy'])
        self.assertTrue(camera.closed)

    def test_full_grasp_without_surface_height_uses_fixed_drop_from_feedback(self):
        cfg, arm, camera, g = self.setup_run((350, 20, 49))
        cfg['work_surface_z_mm'] = None
        move = arm.move

        def move_with_hover_tracking_error(name, pose, fine):
            move(name, pose, fine)
            if name == 'hover':
                arm.current[2] += 2.

        arm.move = move_with_hover_tracking_error
        g.run()
        self.assertEqual([c[0] for c in arm.calls][-4:], ['open', 'descend', 'close', 'lift'])
        np.testing.assert_allclose(g.tcp_for_pose(arm.calls[-3][1]), [350, 20, 37])
        self.assertTrue(camera.closed)

    def test_lost_or_different_target_stops_before_opening(self):
        for point in (None, [410., 20., 45.]):
            _, arm, camera, g = self.setup_run(point)
            with self.assertRaisesRegex(RuntimeError, 'TARGET_LOST_OR_UNSTABLE'):
                g.run()
            self.assertEqual([c[0] for c in arm.calls], ['observe', 'hover'])
            self.assertTrue(camera.closed)

    def test_low_pick_is_rejected_without_opening(self):
        _, arm, camera, g = self.setup_run([350., 20., 15.])
        camera.before_hover[:] = camera.point
        with self.assertRaisesRegex(RuntimeError, 'SURFACE_CLEARANCE'):
            g.run()
        self.assertEqual([c[0] for c in arm.calls], ['observe'])
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
