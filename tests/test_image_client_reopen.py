"""The policy client must be able to reopen its camera after a held switch."""

from unittest import TestCase, mock

from unitree_lerobot.eval_robot.image_server import image_client


class ImageClientReopenTests(TestCase):
    def test_subscriber_manager_reopens_after_camera_close(self):
        manager_type = image_client.ZMQ_SubscriberManager
        with mock.patch.object(manager_type, "_instance", None), \
             mock.patch.object(manager_type, "_subscriber_threads", {}), \
             mock.patch.object(manager_type, "_running", True), \
             mock.patch.object(image_client.zmq, "Context", object):
            old_manager = manager_type.get_instance()
            old_manager.close()

            new_manager = manager_type.get_instance()
            self.assertIsNot(new_manager, old_manager)
            self.assertTrue(new_manager._running)
            new_manager.close()
