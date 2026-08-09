# GR00T N1.7 on G1-29 + Dex3

The deployment keeps three responsibilities separate:

1. The existing TeleImager server on the G1's PC2 publishes the configured head camera.
2. Isaac-GR00T runs the 3B model in its own GPU/Python environment.
3. This `unitree_lerobot` runner executes wherever the robot DDS network is reachable. It reads camera/joint state and, only in an explicitly confirmed mode, publishes arm/Dex3 commands.

The runner can be on the GPU PC if that PC is connected to the robot network. Otherwise it can run on the laptop used for XR teleoperation while the model remains on the GPU PC.

Task text is selected in the Unitree runner and included in every GR00T observation. The model server needs no text-input terminal.

## 1. Keep TeleImager running on PC2

Use the same TeleImager server and `cam_config_server.yaml` used while recording:

```bash
python -m teleimager.image_server --rs
```

The runner requires a fresh response from TeleImager's configuration server, requires `head_camera.enable_zmq` and an advertised 30 FPS, verifies each fresh JPEG against `head_camera.image_shape`, applies the configured binocular `color_0` crop, and converts BGR to RGB. It does not trust the requester's local YAML fallback or silently reuse TeleImager's last decoded BGR frame after a stream timeout. The rolling measured FPS check proves continuing packets only; TeleImager messages do not expose a capture timestamp or sequence number, so exact frame age and sustained 30 FPS still need hardware qualification.

## 2. Start GR00T on the GPU PC

```bash
cd "$HOME/Development/Isaac-GR00T"

CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
    --model-path "$HOME/Development/Models/combined_colour_only_batch_32_acc_1_0908_2/checkpoint-10000" \
    --embodiment-tag NEW_EMBODIMENT \
    --deployment-dataset-path "$HOME/Development/Datasets/lerobot2/combined_atomic_only_09_08_2/train" \
    --host 127.0.0.1 \
    --port 5555
```

`--deployment-dataset-path` reads only `meta/info.json`, and the server requires that path to be the single training dataset recorded inside the selected checkpoint's `experiment_cfg/config.yaml`. It lets the client verify the training data's robot type, 30 Hz control rate, exact ordered 28 joint names, and 480x640 RGB contract. The client also verifies the server's ordered modality keys, history/horizon, and the checkpoint processor contract: this model must decode its relative left/right arm outputs back to absolute joint positions before returning them. A mismatched checkpoint, depth model, different hand ordering, another frame rate, undecoded relative action, or another embodiment fails before command publishers are constructed.

`NEW_EMBODIMENT` alone is not treated as a hardware identity; it is only the generic tag used for this fine-tune.

### If the Unitree runner is on the laptop

Create an encrypted tunnel from the laptop to the GPU PC:

```bash
ssh -N -L 5555:127.0.0.1:5555 alex@GPU_PC_IP
```

The runner still uses `--policy-host 127.0.0.1`; SSH carries that connection to the loopback-only model server. Do not expose the raw GR00T ZMQ port on Wi-Fi. The request contains about 0.9 MB of uncompressed RGB data, so measure tunnel latency and packet loss in shadow mode.

## 3. Run a publisher-free shadow test

Run this on the machine that can reach the G1 DDS network and PC2—normally the XR laptop unless the GPU PC is directly connected:

```bash
cd "$HOME/Development/unitree_lerobot"
# Use the prepared Unitree/TeleImager Python 3.10 environment, then:
python -m pip install -e .

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 192.168.123.164 \
    --network-interface YOUR_ROBOT_NETWORK_INTERFACE \
    --execution-horizon 8 \
    --max-chunks 20
```

Omit `--task` for a menu. The allowlisted IDs preserve the exact training strings:

```text
pick-toothpaste  -> pick up the cylinder toothepaste.
put-toothpaste   -> put down the cylinder toothepaste.
pick-red-cup     -> pick up the red cup.
put-red-cup      -> put down the red cup.
```

Shadow mode constructs state/camera subscribers only; it does not construct DDS command publishers.

## 4. IsaacLab loop verification

This is optional transport and control-loop verification. It is not evidence that the policy learned the simulator scene.

Start the local Unitree IsaacLab G1-29/Dex3 task:

```bash
cd "$HOME/Development/unitree_sim_isaaclab"

python sim_main.py \
    --device cpu \
    --enable_cameras \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --enable_dex3_dds \
    --robot_type g129
```

Then run the same policy loop:

```bash
cd "$HOME/Development/unitree_lerobot"

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --sim \
    --actuate \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 127.0.0.1 \
    --confirm-sim-network-isolated \
    --execution-horizon 8 \
    --max-chunks 20
```

`--sim` selects DDS domain 1 and simulator `rt/lowcmd`, checks the simulator camera config, and converts the simulator's right-hand thumb/middle/index ordering to the recorded thumb/index/middle ordering. The stock simulator and runner both auto-select their CycloneDDS interface; do not pass `--network-interface`. Disconnect or isolate every physical robot network before starting either process, then use `--confirm-sim-network-isolated` as an explicit operator assertion. DDS domain 1 alone is not a physical safety boundary because the simulator reuses robot topic names. The sim path never publishes the real motion-mode `rt/arm_sdk` topic.

## 5. Real actuation is intentionally fail-closed

The current client-side code cannot prove a bounded safe release if DDS itself wedges or the robot loses the publisher. The local Unitree SDK's `Write(timeout=...)` bounds discovery of a matched reader but not the underlying DDS write. The client sends Unitree's documented Dex3 `stopMotors` command during orderly cleanup, but there is no acknowledgment or demonstrated hard deadline. No workstation-only design can send a guaranteed release command over a failed network.

For that reason, real `--actuate` is rejected by default. Before any real test, qualify all of the following on supported hardware with Unitree's normal safety equipment and an operator on the physical emergency stop:

- G1 behavior when the `rt/arm_sdk` publisher disappears or its writes stop;
- Dex3 behavior when both hand command publishers disappear or writes stop;
- the external/robot-side watchdog or lease that handles that failure;
- motion mode ownership, arm tracking/rate limits, and authority ramps;
- absence of every competing XR, replay, or arm/hand DDS publisher.

This adapter accepts only `mode_machine == 5`, corresponding to the current `g1_29dof_with_hand_rev_1_0` asset, and checks it continuously. Mode 2 or another G1 embodiment needs a separately qualified adapter.

After those qualifications, the syntax for an explicitly unqualified research test is shown below so the override cannot be mistaken for a default:

```bash
cd "$HOME/Development/unitree_lerobot"

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 192.168.123.164 \
    --network-interface YOUR_ROBOT_NETWORK_INTERFACE \
    --execution-horizon 8 \
    --max-chunks 1 \
    --actuate \
    --allow-unqualified-real
```

The program still completes a publisher-free observation/inference/action preflight and requires typing `ACTUATE`. On arming, its child process requires fresh state and a stationary 0.5-second dwell, initializes targets from that measured state, and ramps `arm_sdk` weight while holding it. During command execution, state older than 75 ms faults the actuator. These conservative thresholds may need to become stricter after hardware measurement; they are not a substitute for qualification. Orderly SIGINT, SIGTERM, and terminal-hangup cleanup attempts to ramp arm authority back to zero, then sends Unitree's Dex3 `stopMotors` command to both hands; the CLI reports a failed or unacknowledged local release. SIGKILL, power loss, and a wedged DDS/network path can bypass those attempts. The override is not a safety guarantee or certification.
