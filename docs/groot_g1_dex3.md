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

The runner requires a fresh response from TeleImager's configuration server, `head_camera.enable_zmq`, and an advertised 30 FPS. A colour-only checkpoint uses the existing JPEG stream. It verifies each fresh JPEG against `head_camera.image_shape`, applies the configured binocular `color_0` crop, and converts BGR to RGB.

For a checkpoint whose exact video keys are `ego_view, depth_gray_view`, the runner automatically requires the updated atomic RGBD stream:

```yaml
head_camera:
  enable_depth: true
  rgbd_protocol: teleimager-rgbd-v1
  rgbd_zmq_port: 5560
```

Each RGBD packet contains one JPEG and one aligned uint16 depth PNG from the same RealSense capture, with one sequence number. The client rejects corrupt packets, repeated sequences, and a sequence regression caused by a server restart. It reads the live RealSense scale from TeleImager and applies the checkpoint dataset's saved fixed-metric `depth_encoding` (`near_m`, `far_m`, invalid zero and replicated grayscale channels). It does not use `raw_depth_0`, does not encode video during deployment, and does not require a lossless aligned-depth sidecar in the LeRobot dataset.

Neither path trusts the requester's local-YAML fallback or silently reuses a cached frame after transport timeout.

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

For the colour+depth checkpoint, change only `--model-path`:

```bash
--model-path "$HOME/Development/Models/combined_gray_depth_batch_32_acc_1_0908_2/checkpoint-10000"
```

The model's saved modality config selects the camera path automatically. There is no `--depth` or `--grayscale` deployment flag.

`--deployment-dataset-path` reads only `meta/info.json`, and the server requires that path to be the single training dataset recorded inside the selected checkpoint's `experiment_cfg/config.yaml`. It lets the client verify the training data's robot type, 30 Hz control rate, exact ordered 28 joint names, video shapes, and (when selected by the model) saved depth-encoding semantics. The client also verifies the server's ordered modality keys, history/horizon, and the checkpoint processor contract: this model must decode its relative left/right arm outputs back to absolute joint positions before returning them. A mismatched video layout, different hand ordering, another frame rate, unsupported depth encoding, undecoded relative action, or another embodiment fails before command publishers are constructed.

`NEW_EMBODIMENT` alone is not treated as a hardware identity; it is only the generic tag used for this fine-tune.

### If the Unitree runner is on the laptop

Create an encrypted tunnel from the laptop to the GPU PC:

```bash
ssh -N -L 5555:127.0.0.1:5555 alex@GPU_PC_IP
```

The runner still uses `--policy-host 127.0.0.1`; SSH carries that connection to the loopback-only model server. Do not expose the raw GR00T ZMQ port on Wi-Fi. The request contains about 0.9 MB of uncompressed RGB data, so measure tunnel latency and packet loss in shadow mode.

### Synchronous and experimental RTC inference

The default remains `--inference-mode synchronous`: capture one observation, wait for
one prediction, execute its `--execution-horizon` prefix, then repeat. This is the
simpler baseline for comparisons.

`--inference-mode rtc` enables experimental asynchronous **Real-Time Chunking** for a
checkpoint whose configured action horizon is at least 32. The server must advertise
the RTC v1 physical-tail capability; an old server, ReplayPolicy, or wrapped simulator
policy is rejected before DDS is initialized. The exact queue handoff is:

1. After reset/warm-start, make one blocking prediction `A` and start its full horizon
   in the watchdog-owning actuator child.
2. The child advances targets at 30 Hz while continuing to write the current target to
   DDS at 100 Hz.
3. After at least `E = --execution-horizon` actions of the current plan, snapshot its
   exact generation and index `r`, capture a fresh observation, and send the physical
   tail `A[r:]` to GR00T from a separate worker-owned ZeroMQ client.
4. The child continues executing `A` during camera capture, network transfer, model
   inference and decoding. GR00T inpaints a new plan `B`, freezing the estimated
   end-to-end delay prefix and smoothly denoising the remaining overlap.
5. At handoff the child measures `k = current_index - r`, validates the old commanded
   target to `B[k]` boundary, discards `B[:k]`, and atomically continues with `B[k:]`.
6. A stale generation, invalid boundary, failed request, or `k` exhausting the supplied
   overlap is never replayed. The child enters powered HOLD (or the existing actuator
   fault/release path for a hard safety fault).

In RTC mode `--execution-horizon` is the minimum replan spacing and the accounting unit,
not the returned model length. `--max-chunks N` gives an exact action budget of `E*N` at
30 Hz. The automatic frozen prefix estimates recent end-to-end delays. An explicit
`--rtc-frozen-steps` overrides that total budget in 30 Hz actions, including observation
capture, serialization/network, inference, parsing and handoff. Leave
`--rtc-ramp-rate` unset to use the checkpoint's model configuration.

Publisher-free RTC shadow mode runs the same wire protocol against a virtual 30 Hz
action clock. It is useful for latency, stale-response and buffer-budget measurements,
but the real robot does not follow the virtual targets, so it cannot demonstrate
closed-loop smoothness or task quality.

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
pick-toothpaste  -> pick up the cylinder toothpaste.
put-toothpaste   -> put down the cylinder toothpaste.
pick-red-cup     -> pick up the red cup.
put-red-cup      -> put down the red cup.
```

Shadow mode constructs state/camera subscribers only; it does not construct DDS command publishers.

Add `--show-camera` to any shadow, simulation, or real command to display each decoded
`ego_view` frame actually placed in the GR00T observation. RGBD checkpoints also display
`depth_gray_view`. Pressing `q` in a preview window stops the runner through normal cleanup.

Use `--custom-goal "your instruction"` instead of `--task` to send arbitrary language text
to GR00T. The two flags are mutually exclusive. Custom text may be outside the fine-tuning
distribution, so a non-matching instruction requires typing exact uppercase `YES` before
it is sent to the model. The runner retains the normal warm-start, joint-limit, step-size
and tracking checks. Task-bound pose files are unavailable with a custom goal;
use measured or XR-home initialization.

## 4. Initialization and exclusive command ownership

Do not run `xr_teleoperate/teleop/teleop_hand_and_arm.py`, Unitree's policy evaluator, a replay tool, or any other arm/Dex3 command publisher while this client is running. The TeleImager **camera server** on PC2 should remain running; it is not a robot command publisher.

The XR program is already active before its `r/s/q` prompt: it commands all 14 arm joints and both seven-joint hands toward zero. Pressing `q` commands the arms home before exiting, so it is neither a pose-preserving handoff nor an emergency stop. This client therefore owns its complete initialization sequence instead of handing off from XR.

Choose one explicit mode:

- `--initialization measured` is the default. It acquires arm authority while preserving freshly measured arm and hand positions.
- `--initialization xr-home` slowly targets the same joint-zero staging pose used by XR: arms and both hands all zero. Both hands must be empty. This is a staging target, not a demonstrated task-start pose.
- `--initialization pose-file --initial-pose-file PATH` is an experimental route to one reviewed, task-bound demonstration frame-zero pose. It can either preserve both measured hands or explicitly target both hands. Do not use an averaged or median dataset pose.

For example, to request XR-compatible staging, add:

```bash
--initialization xr-home
```

After `ACTUATE`/`SIMULATE`, command authority begins by holding measured positions. A moving initialization then requires typing `INITIALIZE`. Immediately before moving, the actuator again requires a fresh, stationary state at the held target. It follows a fixed, slow 100 Hz smooth joint-space interpolation, keeps its heartbeat/state/mode/tracking checks active, and reports completion only after arm/hand position stability and arm velocity satisfy a continuous dwell. This is the same XR joint-zero **target**, with a deliberately slower guarded motion profile. The interpolation is not collision-aware, so a clear workspace, support and an emergency-stop operator remain mandatory. Finally, the operator visually checks the robot and scene and types `RUN`. Only then does the client reset GR00T, capture fresh state/images, run new inference, apply executable step checks, and submit a policy action. The earlier publisher-free model result is never executed.

Actuated runs use policy warm-start by default. After `RUN`, the client performs one fresh inference while deliberately skipping only the current-pose-to-first-target jump check. It still validates the full response, joint limits, finiteness, and within-chunk step sizes. After the operator types `WARMUP`, the actuator follows its existing bounded 100 Hz interpolation to that first target. The entire inferred chunk is then discarded. After `CONTINUE`, the client resets GR00T, captures a new observation from the reached pose, and restores the normal 0.05-rad arm and 0.10-rad hand action-step ceilings for every executed chunk. Use `--no-policy-warm-start` only when deliberately testing the strict direct-start behavior.

This transition is available in both IsaacLab and the explicitly unqualified real path. It is a per-goal joint-space transition, not collision-aware planning and not a general license for large policy jumps. On the real robot, `WARMUP` and `CONTINUE` are separate confirmations so the operator can inspect the target delta and the reached pose.

During a synchronous actuated run, terminal commands are read at safe action-chunk
boundaries. RTC polls them continuously while its child owns the exact 30 Hz handoff:

- `hold` or `h`, followed by Enter, captures the current measured arm/hand pose and keeps publishing it at 100 Hz. The client stops making GR00T requests and displays a next-goal prompt. This is a powered position hold, not a passive brake or collision-safe freeze; it can continue exerting force and requires the client/watchdog to remain alive.
- At the HOLD prompt, enter a trained task ID, its menu number, its exact training sentence, or custom goal text. Unknown text again requires exact `YES`. The new goal repeats GR00T reset, fresh inference, `WARMUP`, discarded chunk, `CONTINUE`, reset, and fresh strict inference.
- `quit` or `q` requests orderly release and exits. `Ctrl-C` also enters cleanup from any state. On real hardware, cleanup retains the final arm target while ramping `arm_sdk` authority to zero over 1.5 seconds, then sends Dex3 `stopMotors`; final pose and grasp after Unitree retakes authority are not guaranteed.

A command typed while a synchronous inference request is already running cannot cancel that network request. Its result is discarded before submission, then HOLD or release occurs. In RTC, `h` interrupts the active plan into powered HOLD and drains/discards any in-flight reply before accepting a new goal; `q` begins orderly release without waiting for the reply. Reaching finite `--max-chunks` without a HOLD command still exits through normal release; use a sufficiently large but finite bound for an interactive pick/put session.

Armed `INITIALIZE`, `RUN`, `WARMUP`, and `CONTINUE` prompts time out after 60 seconds. The HOLD goal prompt may wait indefinitely while continuing the heartbeat and safety checks. `Ctrl-C`, rejection, timeout or a watchdog fault enters the existing authority-release and Dex3 `stopMotors` cleanup path.

A pose file has this strict schema (all 28 `joint_names` must appear in the exact training order). The structure below is deliberately non-runnable: export and review one real episode's frame-zero values before replacing the arm placeholder.

```json
{
  "schema_version": 1,
  "name": "replace with a reviewed red-cup pick frame-zero pose",
  "robot_type": "Unitree_G1_Dex3_HeadOnly",
  "task": "pick-red-cup",
  "instruction": "pick up the red cup.",
  "joint_names": [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"
  ],
  "arm": ["REPLACE_WITH_14_REVIEWED_NUMERIC_VALUES"],
  "hands": {"policy": "measured"},
  "source": {
    "dataset_path": "/absolute/path/to/the/training/dataset",
    "episode_index": 0,
    "frame_index": 0
  }
}
```

To move the hands too, replace `hands` with `{"policy": "explicit", "left": [seven values], "right": [seven values]}`. The loader rejects extra/missing fields, a different selected task or instruction, the wrong robot/joint order, non-frame-zero provenance, non-numeric values, and targets outside the deployment joint limits. `source` is operator-supplied audit provenance; this version validates its shape but cannot cryptographically prove the pose values came from that episode, so the file still requires human review.

## 5. IsaacLab loop verification

This is optional transport and control-loop verification. It is not evidence that the policy learned the simulator scene.

The current IsaacLab TeleImager path supplies colour only. Use the colour-only checkpoint for this loop unless the simulator camera server is separately extended to publish the same atomic aligned-depth contract.

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

After the complete synchronous ladder passes, a 32-action checkpoint can exercise the
same isolated simulator with:

```bash
python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --sim \
    --actuate \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 127.0.0.1 \
    --confirm-sim-network-isolated \
    --inference-mode rtc \
    --execution-horizon 8 \
    --max-chunks 20
```

Keep automatic RTC delay estimation for the first test and record every request index,
overlap, frozen prefix, inference time, actual child-measured delay and handoff. An RTC
buffer underrun must produce powered HOLD, never replay or silently fall back to
independent chunks.

`--sim` selects DDS domain 1 and simulator `rt/lowcmd`, checks the simulator camera config, and converts the simulator's right-hand thumb/middle/index ordering to the recorded thumb/index/middle ordering. The stock simulator and runner both auto-select their CycloneDDS interface; do not pass `--network-interface`. Disconnect or isolate every physical robot network before starting either process, then use `--confirm-sim-network-isolated` as an explicit operator assertion. DDS domain 1 alone is not a physical safety boundary because the simulator reuses robot topic names. The sim path never publishes the real motion-mode `rt/arm_sdk` topic.

## 6. Real actuation is intentionally fail-closed

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
    --initialization xr-home \
    --actuate \
    --allow-unqualified-real
```

The program still completes a publisher-free observation/inference/action preflight and requires typing `ACTUATE`, then `INITIALIZE` for the moving XR-home request, and finally `RUN`. The default warm-start additionally requires `WARMUP` before each new goal's transition and `CONTINUE` before fresh strict inference. On arming, its child process requires fresh state and a stationary 0.5-second dwell, initializes targets from that measured state, and ramps `arm_sdk` weight while holding it. During command execution, state older than 75 ms faults the actuator. These conservative thresholds may need to become stricter after hardware measurement; they are not a substitute for qualification. Orderly SIGINT, SIGTERM, and terminal-hangup cleanup attempts to ramp arm authority back to zero, then sends Unitree's Dex3 `stopMotors` command to both hands; the CLI reports a failed or unacknowledged local release. SIGKILL, power loss, and a wedged DDS/network path can bypass those attempts. The override is not a safety guarantee or certification.
