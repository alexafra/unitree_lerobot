"""Offline action-debug checks: fake publishers and a filesystem-only writer."""

from collections import deque
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.action_debug import ActionDebugRecorder, ActionDebugSampler, ActionDebugSink
from unitree_lerobot.eval_robot.g1_end_effectors import DEX3_PROFILE, INSPIRE_FTP_PROFILE
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    ARM_JOINT_NAMES, PublishedHandTarget, RobotState, _G1Dex3CommandBackend,
)


def make_sink(work_queue=None):
    counters = [0, 0, 0, 0, 0]
    accepting = SimpleNamespace(value=1)
    alive = SimpleNamespace(value=1)
    sink = ActionDebugSink("actuator", work_queue if work_queue is not None else queue.Queue(256),
                           counters, accepting, alive)
    return sink, counters, accepting, alive


class ActionDebugSinkTests(unittest.TestCase):
    def test_full_queue_drops_immediately_without_json_or_filesystem_work(self):
        sink, counters, _, _ = make_sink(queue.Queue(1))
        self.assertTrue(sink.record("first", action=np.zeros(26)))
        started = time.monotonic()
        with mock.patch("unitree_lerobot.eval_robot.action_debug.json.dumps", side_effect=AssertionError), \
                mock.patch.object(Path, "open", side_effect=AssertionError):
            for _ in range(1000):
                self.assertFalse(sink.record("overflow", action=np.zeros(26)))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(counters[:3], [1001, 1000, 0])

    def test_arrays_are_copied_before_async_feeder_serialization(self):
        sink, _, _, _ = make_sink()
        source = np.arange(6, dtype=np.float32)
        sink.record("plan", nested={"actions": [source]})
        source[:] = -1
        item = sink._queue.get_nowait()
        np.testing.assert_array_equal(item["fields"]["nested"]["actions"][0], np.arange(6))

    def test_broken_queue_and_dead_writer_cannot_raise_to_control(self):
        broken = mock.Mock()
        broken.put_nowait.side_effect = OSError("closed pipe")
        sink, counters, _, alive = make_sink(broken)
        self.assertFalse(sink.record("broken"))
        self.assertEqual(counters[:3], [1, 1, 1])
        alive.value = 0
        self.assertFalse(sink.record("writer_dead"))
        self.assertEqual(broken.put_nowait.call_count, 1)

    def test_detaching_never_waits_for_or_joins_the_feeder(self):
        work_queue = mock.Mock()
        work_queue.cancel_join_thread.side_effect = OSError("already closed")
        sink, _, _, _ = make_sink(work_queue)
        sink.detach_at_process_exit()
        work_queue.join_thread.assert_not_called()


class ActionDebugWriterTests(unittest.TestCase):
    def test_disabled_recorder_does_not_create_queue_process_or_files(self):
        with mock.patch("unitree_lerobot.eval_robot.action_debug.mp.get_context", side_effect=AssertionError):
            recorder = ActionDebugRecorder("/must/not/be/created", enabled=False)
            self.assertIsNone(recorder.make_sink())
            self.assertFalse(recorder.record("ignored", target=np.zeros(26)))
            self.assertFalse(recorder.close()["enabled"])

    def test_writer_serializes_arrays_in_new_private_directory_and_closes_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "action_debug"
            recorder = ActionDebugRecorder(output, metadata={"profile": "inspire-ftp"})
            self.assertTrue(recorder.record("policy_plan", arm=np.arange(14), left=np.ones(6) * 0.5))
            first = recorder.close(timeout_s=5.0)
            self.assertEqual(first, recorder.close())
            self.assertEqual(first["written"], 1, first)
            self.assertFalse(first["writer_failed"], first)
            record = json.loads((output / "events.jsonl").read_text().strip())
            self.assertEqual(record["fields"]["arm"], list(range(14)))
            self.assertEqual(record["fields"]["left"], [0.5] * 6)
            self.assertEqual((output / "events.jsonl").stat().st_mode & 0o777, 0o600)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["clean_shutdown"])

    def test_existing_directory_is_not_overwritten_and_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "events.jsonl"
            existing.write_text("keep this")
            recorder = ActionDebugRecorder(temporary)
            recorder.record("ignored")
            summary = recorder.close(timeout_s=5.0)
            self.assertTrue(summary["writer_failed"])
            self.assertEqual(existing.read_text(), "keep this")

    def test_serialization_failure_drops_one_record_without_losing_valid_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ActionDebugRecorder(Path(temporary) / "debug")
            recorder.record("invalid", target=float("nan"))
            recorder.record("valid", target=0.5)
            summary = recorder.close(timeout_s=5.0)
            self.assertEqual(summary["serialization_dropped"], 1, summary)
            self.assertEqual(summary["written"], 1, summary)
            self.assertFalse(summary["writer_failed"], summary)

    def test_main_and_inference_threads_have_independent_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ActionDebugRecorder(Path(temporary) / "debug", queue_capacity=256)
            def emit():
                for index in range(25):
                    recorder.record("policy_request", index=index)
            worker = threading.Thread(target=emit)
            worker.start()
            emit()
            worker.join()
            summary = recorder.close(timeout_s=5.0)
            self.assertEqual(summary["written"], 50, summary)
            self.assertEqual(summary["sources"]["policy"]["submitted"], 25)
            self.assertEqual(summary["sources"]["policy-thread-1"]["submitted"], 25)
            records = [json.loads(line) for line in (Path(temporary) / "debug/events.jsonl").read_text().splitlines()]
            identities = {(record["source"], record["source_index"]) for record in records}
            self.assertEqual(len(identities), 50)

    def test_writer_start_failure_is_fail_isolated(self):
        with mock.patch("unitree_lerobot.eval_robot.action_debug.mp.get_context", side_effect=OSError("unavailable")):
            recorder = ActionDebugRecorder("/not/created")
            self.assertIsNone(recorder.make_sink())
            self.assertFalse(recorder.record("ignored"))
            self.assertIn("unavailable", recorder.close()["error"])

    def test_terminal_summary_includes_missed_sampling_slots_even_if_events_drop(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ActionDebugRecorder(Path(temporary) / "debug")
            recorder.make_sink("actuator").note_sampling(25, 2)
            summary = recorder.close(timeout_s=5.0)
            self.assertEqual(summary["sources"]["actuator"]["missed_sample_slots"], 25)
            self.assertEqual(summary["sources"]["actuator"]["sample_errors"], 2)

    def test_abrupt_writer_death_only_drops_data_and_close_remains_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ActionDebugRecorder(Path(temporary) / "debug", queue_capacity=1)
            recorder._process.kill()
            recorder._process.join(timeout=1.0)
            started = time.monotonic()
            for _ in range(100):
                recorder.record("after_writer_crash", target=np.zeros(26))
            summary = recorder.close(timeout_s=0.1)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(summary["writer_failed"])
            self.assertGreater(summary["sources"]["policy"]["queue_dropped"], 0)

    def test_thread_sink_reuse_supports_more_than_sixteen_goal_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ActionDebugRecorder(Path(temporary) / "debug")
            def emit():
                try:
                    recorder.record("worker_response", action=np.zeros(26))
                finally:
                    recorder.release_thread_sink()
            for _ in range(40):
                worker = threading.Thread(target=emit)
                worker.start()
                worker.join()
            summary = recorder.close(timeout_s=5.0)
            self.assertEqual(summary["written"], 40, summary)
            self.assertEqual(summary["untracked_thread_dropped"], 0, summary)
            records = [json.loads(line) for line in (Path(temporary) / "debug/events.jsonl").read_text().splitlines()]
            self.assertEqual(len({(row["source"], row["source_index"]) for row in records}), 40)


class ActionDebugSamplerTests(unittest.TestCase):
    def make_backend(self, profile=INSPIRE_FTP_PROFILE):
        state = RobotState(1.0, 5, np.arange(14) * 0.01, np.arange(14) * 0.02,
                           np.ones(profile.hand_dof) * 0.4, np.ones(profile.hand_dof) * 0.6,
                           arm_received_at=0.99, left_hand_received_at=0.98, right_hand_received_at=0.97)
        backend = SimpleNamespace(
            _arm_target=np.ones(14) * 0.2, _left_target=np.ones(profile.hand_dof) * 0.3,
            _right_target=np.ones(profile.hand_dof) * 0.7,
            _last_published_arm_q=np.ones(14) * 0.19, _last_published_arm_tau=np.zeros(14),
            _left_hand_publish_history=deque([PublishedHandTarget(1.1, np.ones(profile.hand_dof) * 0.31)]),
            _right_hand_publish_history=deque([PublishedHandTarget(1.2, np.ones(profile.hand_dof) * 0.71)]),
            _action_debug_arm_completed_at=1.05, _weight=1.0, _last_publish_timing_ms={"arm_write": 0.5},
        )
        return backend, state

    def test_profile_units_raw_desired_conditioned_published_and_feedback_remain_distinct(self):
        for profile in (DEX3_PROFILE, INSPIRE_FTP_PROFILE):
            sink, _, _, _ = make_sink()
            sampler = ActionDebugSampler(sink, profile, ARM_JOINT_NAMES)
            backend, sampler.state = self.make_backend(profile)
            sampler.context = {"sequence": 4, "next_action_index": 3, "rtc": True}
            sampler.scheduled(np.ones(14) * 0.5, np.ones(profile.hand_dof), np.zeros(profile.hand_dof),
                              sequence=4, action_index=2, scheduled_at=1.0)
            sampler.desired = {"arm": np.ones(14) * 0.45}
            sampler.sample(backend, now=1.3)
            metadata, scheduled, sample = [sink._queue.get_nowait() for _ in range(3)]
            self.assertEqual(metadata["fields"]["units"]["hand_position"], profile.value_unit)
            self.assertEqual(metadata["fields"]["units"]["arm_velocity"], "rad/s")
            fields = sample["fields"]
            self.assertEqual(fields["context"]["sequence"], 4)
            np.testing.assert_array_equal(fields["measured"]["arm_dq"], sampler.state.arm_dq)
            self.assertEqual(fields["raw_scheduled_target"]["arm"][0], 0.5)
            self.assertEqual(fields["desired_target"]["arm"][0], 0.45)
            self.assertEqual(fields["conditioned_target"]["arm"][0], 0.2)
            self.assertEqual(fields["published"]["arm"][0], 0.19)
            self.assertEqual(fields["published"]["left_hand"][0], 0.31)
            self.assertEqual(fields["published"]["right_hand_completed_at"], 1.2)
            self.assertEqual(scheduled["fields"]["action_index"], 2)

    def test_sampling_is_bounded_and_missing_slots_are_counted_without_replay(self):
        sink, counters, _, _ = make_sink()
        sampler = ActionDebugSampler(sink, INSPIRE_FTP_PROFILE, ARM_JOINT_NAMES)
        backend, sampler.state = self.make_backend()
        for tick in range(100):
            sampler.sample(backend, now=tick / 100)
        self.assertLessEqual(counters[0], 32)  # one metadata record plus <=31 samples
        before = counters[0]
        sampler.sample(backend, now=4.0)
        self.assertEqual(counters[0], before + 1)
        self.assertGreater(sampler.missed_samples, 80)
        sampler.finish()
        self.assertEqual(counters[3], sampler.missed_samples)

    def test_diagnostic_failure_cannot_suppress_or_duplicate_publication(self):
        backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
        backend._publish_arm = mock.Mock()
        backend._publish_hands = mock.Mock()
        backend._action_debug_sampler = mock.Mock()
        backend._action_debug_sampler.sample.side_effect = RuntimeError("debug broke")
        backend.publish()
        backend._publish_arm.assert_called_once()
        backend._publish_hands.assert_called_once()
        backend._action_debug_sampler.sample.assert_called_once_with(backend, publish_ok=True)

    def test_original_publication_error_survives_diagnostic_failure(self):
        backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
        backend._publish_arm = mock.Mock(side_effect=RuntimeError("DDS write failed"))
        backend._publish_hands = mock.Mock()
        backend._action_debug_sampler = mock.Mock()
        backend._action_debug_sampler.sample.side_effect = ValueError("debug broke")
        with self.assertRaisesRegex(RuntimeError, "DDS write failed"):
            backend.publish()
        backend._publish_hands.assert_not_called()
        backend._action_debug_sampler.sample.assert_called_once_with(backend, publish_ok=False)

    def test_backend_publication_default_has_no_debug_side_effects(self):
        backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
        backend._publish_arm = mock.Mock()
        backend._publish_hands = mock.Mock()
        with mock.patch("unitree_lerobot.eval_robot.action_debug.ActionDebugSink.record", side_effect=AssertionError):
            backend.publish()
        backend._publish_arm.assert_called_once()
        backend._publish_hands.assert_called_once()
        self.assertFalse(hasattr(backend, "_action_debug_full_publish"))


if __name__ == "__main__":
    unittest.main()
