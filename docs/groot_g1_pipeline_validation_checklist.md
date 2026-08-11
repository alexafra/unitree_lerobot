# Informal GR00T → Unitree G1/Dex3 Pipeline Validation Checklist

Last updated: 2026-08-10

This is a living engineering checklist, not a safety certification. It separates model
quality, software-contract correctness, simulator behavior, and physical-robot safety.
Passing an earlier gate does not imply that a later gate is safe.

The intended first-deployment topology is:

```text
G1 PC2: TeleImager camera server only
             |
             v
Unitree runner: camera/state adapter + DDS actuator/watchdog
             |
             | localhost, or an SSH localhost tunnel
             v
GPU PC: Isaac-GR00T PolicyServer + final checkpoint
```

The Unitree runner must be the only arm/Dex3 command owner. Do not run
`teleop_hand_and_arm.py`, Unitree `eval_g1.py`, replay tools, or another arm/hand DDS
publisher at the same time. The TeleImager camera server should remain running.

## Status legend

- `[ ]` Not yet demonstrated.
- `[x]` Demonstrated, with the evidence path/date written beside it.
- `STOP` A hard gate. Do not progress until it passes.

## Validation ladder

```text
Freeze contract and source
    → automated tests
    → final offline evaluation
    → server contract smoke test
    → publisher-free hardware shadow soak
    → isolated IsaacLab command loop
    → robot-side failure qualification
    → measured hold only
    → XR-home initialization only
    → one policy step
    → one short chunk
    → bounded multi-chunk task trial
```

---

## 0. Freeze the final deployment contract

For the first physical deployment, prefer the colour-only model. Validate RGBD later as
a separate change. Keep the existing upper-body `NEW_EMBODIMENT` contract unless there is
a deliberate reason to change the adapter:

- 30 Hz dataset/control semantics.
- `ego_view`, shape `480x640x3`, RGB `uint8`.
- One observation timestep: `delta_indices=[0]`.
- State/action keys, in order: `left_arm`, `right_arm`, `left_hand`, `right_hand`.
- Exact 28 joint names and ordering already recorded in `meta/info.json`.
- Relative model representation for the two arms, decoded by GR00T to absolute positions.
- Absolute model representation for both hands.
- Predicted action horizon is checkpoint-specific. Both current 20k colour-only and
  gray-depth runs use 32; older colour checkpoints may still expose 16. Deployment
  `--execution-horizon` may use any positive prefix up to the checkpoint's advertised
  action horizon.
- Exact task strings in the runner allowlist.

The runtime now has two deliberately separate modes. `synchronous` remains the baseline.
Experimental `rtc` requires a predicted horizon of at least 32 and an RTC-capable direct
PyTorch server. Treat it as a separate change: complete the synchronous simulator and
physical ladder first, then repeat the relevant gates for RTC rather than changing the
model, scene, network and scheduler together.

Record the final paths once selected:

```bash
FINAL_RUN_DIR="$HOME/Development/Models/REPLACE_FINAL_RUN"
FINAL_STEP=REPLACE_FINAL_STEP
FINAL_CHECKPOINT="$FINAL_RUN_DIR/checkpoint-$FINAL_STEP"
TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/REPLACE_FINAL_DATASET/train"
VALIDATION_DATASET="/home/alex/Development/Datasets/lerobot2/REPLACE_FINAL_DATASET/validation"
ROBOT_NIC="REPLACE_WITH_EXPLICIT_ROBOT_INTERFACE"
INITIAL_POSE_FILE="/absolute/path/to/a/reviewed/task_frame0_pose.json"
```

- [ ] Final colour/RGBD choice recorded: ____________________
- [ ] Final checkpoint: ____________________________________
- [ ] Training dataset: ____________________________________
- [ ] Validation dataset: __________________________________
- [ ] Task strings reviewed against `meta/tasks.jsonl`.
- [ ] No unplanned change to FPS, joint order, action representation, or image history.

`STOP`: If any of those contract fields change, update and retest the deployment adapter
before starting the server. The client is intentionally fail-closed on mismatches.

---

## 1. Freeze and identify the exact source revisions

The current deployment changes in `unitree_lerobot` should be reviewed and committed (or
otherwise archived with hashes) before physical testing. Do not deploy from an unknown,
changing worktree.

```bash
git -C "$HOME/Development/Isaac-GR00T" status --short
git -C "$HOME/Development/Isaac-GR00T" rev-parse HEAD

git -C "$HOME/Development/unitree_lerobot" status --short
git -C "$HOME/Development/unitree_lerobot" rev-parse HEAD
git -C "$HOME/Development/unitree_lerobot" diff --check

git -C "$HOME/Development/xr_teleoperate" status --short
git -C "$HOME/Development/xr_teleoperate" rev-parse HEAD
```

Record these artifacts with every deployment run:

```bash
sha256sum \
    "$FINAL_CHECKPOINT/processor_config.json" \
    "$FINAL_CHECKPOINT/experiment_cfg/config.yaml" \
    "$TRAIN_DATASET/meta/info.json" \
    "$TRAIN_DATASET/meta/modality.json"
```

- [ ] Isaac-GR00T revision recorded: ________________________
- [ ] unitree_lerobot revision recorded: ____________________
- [ ] XR/TeleImager revision recorded: ______________________
- [ ] Python environment/lock or installed-package snapshot recorded.
- [ ] G1 firmware, robot asset/revision and `mode_machine` recorded.
- [ ] PC2 camera configuration and camera identity recorded.
- [ ] Final checkpoint/dataset hashes saved with run records.

---

## 2. Run the software-only regression tests

### Isaac-GR00T server/contract tests

```bash
cd "$HOME/Development/Isaac-GR00T"

# If this checkout has not installed the locked development/test dependencies:
# uv sync --all-extras

uv run --no-sync python -m pytest \
    tests/gr00t/eval/test_server_modality_json.py \
    tests/gr00t/model/test_action_head_rtc.py \
    tests/gr00t/policy/test_gr00t_policy.py \
    tests/gr00t/policy/test_policy_service.py \
    -q
```

### Unitree deployment tests

Use the prepared Unitree/TeleImager Python 3.10 environment:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot

python -m unittest discover \
    -s tests \
    -p 'test_groot_g1_deployment.py' \
    -v

python -m unittest discover \
    -s tests \
    -p 'test_groot_g1_rtc.py' \
    -v

python -m unittest discover \
    -s tests \
    -p 'test_depth_encoding.py' \
    -v

python -m unitree_lerobot.eval_robot.eval_groot_g1 --help
```

### Cross-codebase serializer/contract test

Run the Unitree deployment tests once with Isaac-GR00T's Python 3.12 environment on the
same import path. This specifically exercises the real GR00T serializer compatibility:

```bash
cd "$HOME/Development/Isaac-GR00T"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$HOME/Development/unitree_lerobot:$HOME/Development/Isaac-GR00T" \
.venv/bin/python -m unittest discover \
    -s "$HOME/Development/unitree_lerobot/tests" \
    -p 'test_groot_g1_deployment.py' \
    -v
```

- [ ] All runnable tests pass.
- [ ] Only understood dependency-related skips remain.
- [ ] No import error in the actual Python 3.10 runtime that will run the robot client.

`STOP`: Mocks and unit tests validate code paths, not DDS failure behavior or physical
joint directions. They are necessary but never sufficient for hardware actuation.

---

## 3. Train and select the final checkpoint

- [ ] Compute dataset statistics using the exact final modality config.
- [ ] Train from the intended N1.7 base checkpoint.
- [ ] Confirm the checkpoint contains:
  - `processor_config.json`;
  - `experiment_cfg/config.yaml` recording exactly one training dataset for
    `new_embodiment`;
  - processor/model files required by `Gr00tPolicy`.
- [ ] Select the checkpoint using held-out validation, not training loss alone.
- [ ] Retain the complete training/evaluation logs.

### Final comprehensive open-loop evaluation

Run only the selected checkpoint, over complete validation episodes:

```bash
cd "$HOME/Development/Isaac-GR00T"

CUDA_VISIBLE_DEVICES=0 \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
    --run-dir "$FINAL_RUN_DIR" \
    --dataset-path "$VALIDATION_DATASET" \
    --train-dataset-path "$TRAIN_DATASET" \
    --output-dir "$FINAL_RUN_DIR/evaluation_final_h8" \
    --checkpoint-steps "$FINAL_STEP" \
    --steps 0 \
    --execution-horizon 8 \
    --denoising-steps 4 \
    --modality-keys left_arm right_arm left_hand right_hand \
    --train-probe-episodes 3 \
    --train-probe-seed 42
```

Confirm that the output includes the compressed frame-level data:

```text
evaluation_final_h8/checkpoint-<step>/validation_frame_predictions.csv.gz
evaluation_final_h8/checkpoint-<step>/train_probe_frame_predictions.csv.gz
```

Before the full run, representative episodes may be compared at execution horizons 1, 2,
4, and 8 using separate output directories and explicit `--traj-ids`.

- [ ] No NaN/Inf or malformed output.
- [ ] No predicted value outside physical limits.
- [ ] Inspect every arm/hand joint, not aggregate MAE alone.
- [ ] Inspect first-step discontinuities and the large right-hand spikes previously seen.
- [ ] Validation metrics are plausible relative to the fixed training probe.
- [ ] The chosen horizon is recorded with the model.

Open-loop MAE/MSE does not prove closed-loop task success or safe execution.

---

## 4. Start the final policy server and prove its contract

Run the server on the GPU PC. Keep it loopback-only:

```bash
cd "$HOME/Development/Isaac-GR00T"

CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync python -m gr00t.eval.run_gr00t_server \
    --model-path "$FINAL_CHECKPOINT" \
    --embodiment-tag NEW_EMBODIMENT \
    --deployment-dataset-path "$TRAIN_DATASET" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port 5555
```

Do not pass a training-time modality config during inference. The checkpoint's saved
processor config is the source of truth.

- [ ] Server reaches `Server ready` without dataset-path mismatch.
- [ ] Selected video keys are correct for the checkpoint.
- [ ] Action-output metadata says absolute joint positions are returned.
- [ ] Server is bound to `127.0.0.1`, not exposed unauthenticated on Wi-Fi/LAN.

If the Unitree runner is on the established XR laptop rather than the GPU PC, create an
SSH tunnel from that laptop:

```bash
ssh -N -L 5555:127.0.0.1:5555 alex@GPU_PC_IP
```

The runner still connects to `--policy-host 127.0.0.1`.

---

## 5. Establish one deployment topology

Preferred choices, in order:

1. GPU PC runs server and runner as separate processes/environments, but only after its
   dedicated wired NIC is proven to receive G1 DDS state and reach PC2 TeleImager.
2. The established XR laptop runs the Unitree runner and reaches the GPU server through
   the SSH localhost tunnel.

Do not put the full Python runner on PC2 for the first deployment. PC2 runs TeleImager.

- [ ] Client host selected: GPU PC / laptop
- [ ] Robot DDS interface name recorded: ____________________
- [ ] PC2 is reachable over the intended wired network.
- [ ] Shadow mode proves DDS state reception; `ping` alone is insufficient.
- [ ] No raw policy traffic crosses Wi-Fi without an SSH tunnel.

Before every command-capable run, inspect for competing processes:

```bash
pgrep -af 'teleop_hand_and_arm|eval_g1.py|replay_robot|eval_groot_g1'
```

The only permitted match during a GR00T run is the one intended
`eval_groot_g1` process. Check other computers and services too; a local process listing
cannot prove there is no remote DDS writer.

---

## 6. Validate TeleImager and physical state in read-only mode

On G1 PC2, use the exact TeleImager configuration used for data collection:

```bash
python -m teleimager.image_server --rs
```

For the first deployment, use the colour-only checkpoint and existing JPEG stream.

- [ ] Fresh camera-config response arrives from PC2, not the local YAML fallback.
- [ ] Advertised stream is 480x640 at 30 FPS.
- [ ] The live image has the same crop, orientation, colour order, and viewpoint as the
  training `ego_view`.
- [ ] No cached image is accepted after stopping the camera stream.
- [ ] G1 DDS `mode_machine` is observed and recorded without creating publishers.
- [ ] Real measured arm/hand values are finite and inside the configured limits.
- [ ] Verify the actual 14 arm and 14 hand channel ordering against the training names.
  Unit tests prove the intended mapping; this step proves the connected hardware.

`STOP`: Do not infer joint ordering merely from matching array lengths.

---

## 7. Run a publisher-free shadow smoke test

On the selected Unitree runner host, using its Python 3.10 environment:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot
python -m pip install -e .

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --execution-horizon 8 \
    --max-chunks 1 \
    2>&1 | tee shadow_smoke.log
```

- [ ] Server `ping` succeeds.
- [ ] Dataset, modality, action semantics, image, FPS and joint-order contracts pass.
- [ ] Log says `SHADOW MODE: no command publishers were created`.
- [ ] An independent DDS/topic observation confirms that shadow mode creates no writer on
  `rt/arm_sdk` or either Dex3 command topic; do not rely only on the program's own log.
- [ ] One full observation → inference → action validation pass succeeds.
- [ ] First-step arm/hand deltas are recorded and plausible for the actual starting pose.

Repeat once for every trained task in a scene/start state appropriate to that task. A
prediction rejected from an unrelated starting pose is not automatically a transport bug.

---

## 8. Run a shadow latency and freshness soak

Run at least 100 publisher-free chunks initially, then a 500-chunk final soak for the
intended topology and final checkpoint:

```bash
python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --execution-horizon 8 \
    --max-chunks 100 \
    2>&1 | tee shadow_soak_100.log
```

For the final soak, change `--max-chunks 100` to `--max-chunks 500`, use a new log, and
repeat across all four allowlisted tasks in appropriate scenes.

Review inference times and calculate p50/p95/p99/maximum from the saved log:

```bash
rg -o 'inference [0-9.]+s' shadow_soak_100.log
```

The live actuator heartbeat expires after 1.0 second. Inference is intentionally not
heartbeated, so a model/network stall releases rather than hiding the stall. The warmed
worst case must remain comfortably below 1.0 second; use 0.75–0.80 seconds as a provisional
engineering margin until the timing budget is measured and qualified.

- [ ] No server timeout or socket recovery during the soak.
- [ ] No stale DDS state or stale/repeated camera frame.
- [ ] No action shape, range or step-limit rejection.
- [ ] Warmed P50 inference: __________ s
- [ ] Warmed P95 inference: __________ s
- [ ] Warmed P99 inference: __________ s
- [ ] Warmed maximum inference: ______ s
- [ ] Provisional gate: p99 <= 0.50 s and no warmed sample > 0.75 s. Replace these with a
  reviewed measured timing budget if the architecture changes.
- [ ] Repeat over the actual SSH tunnel if the laptop topology will be used.

`STOP`: If inference or transport can exceed the heartbeat deadline, do not weaken the
watchdog merely to make the run continue. Fix/accelerate the pipeline or redesign the
execution/watchdog architecture first.

---

## 9. Validate the command loop in isolated IsaacLab

Use the colour-only model. The current IsaacLab camera path does not provide the physical
atomic RGBD contract.

Physically disconnect or otherwise isolate every real-robot network from the simulator
host. DDS domain 1 alone is not isolation because simulator and robot topics overlap.

Start the simulator:

```bash
cd "$HOME/Development/unitree_sim_isaaclab"
conda activate unitree_sim_env

python sim_main.py \
    --device cpu \
    --enable_cameras \
    --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
    --enable_dex3_dds \
    --robot_type g129
```

First prove measured hold and cancellation without a policy action:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --sim \
    --actuate \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 127.0.0.1 \
    --confirm-sim-network-isolated \
    --initialization measured \
    --execution-horizon 1 \
    --max-chunks 1
```

Press `r` at the SIMULATE gate; at the RUN gate press Ctrl-C. Next change to
`--initialization xr-home`, press `r` at the SIMULATE and INITIALIZE gates, then press
Ctrl-C at RUN. Only after both cleanup paths pass should the simulator execute a policy
action by pressing `r` at RUN. None of these gates requires Enter.

Then expand deliberately:

- [ ] `execution-horizon=1`, `max-chunks=1`.
- [ ] `execution-horizon=2`, `max-chunks=1`.
- [ ] `execution-horizon=4`, `max-chunks=1`.
- [ ] `execution-horizon=8`, `max-chunks=1`.
- [ ] `execution-horizon=8`, `max-chunks=20`.
- [ ] For a 32-action checkpoint, `execution-horizon=16`, `max-chunks=20`.

Simulator pass criteria:

- [ ] No command before explicit confirmation.
- [ ] First commanded target begins at the measured hold pose.
- [ ] Arm joint directions and right-hand permutation are correct.
- [ ] XR-home motion is smooth and bounded.
- [ ] Action targets advance at 30 Hz while DDS publishing continues at 100 Hz.
- [ ] Ctrl-C causes orderly release reporting.
- [ ] In separate isolated tests, stopping the policy server, camera, and DDS state each
  causes a fault rather than stale-action continuation.
- [ ] No attempt is made to claim task success from a scene that does not match training.

### Separate experimental RTC gate

Do this only with a checkpoint exposing at least 32 actions and only after every
synchronous item above passes. First run publisher-free RTC shadow with a large finite
budget. Its virtual clock tests the physical-tail wire protocol and end-to-end timing;
it does not test closed-loop behavior because no predicted action moves the robot.

```bash
python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 127.0.0.1 \
    --inference-mode rtc \
    --execution-horizon 8 \
    --max-chunks 100
```

Then repeat in isolated IsaacLab with `--sim --actuate
--confirm-sim-network-isolated`. Keep automatic frozen-delay estimation initially.

- [ ] Server metadata advertises RTC protocol v1, physical action tails and direct
  PyTorch backend; an old/replay/wrapped server fails before DDS initialization.
- [ ] Shadow logs every request origin, overlap, frozen prefix, end-to-end inference
  duration, child/virtual elapsed action count and handoff.
- [ ] `execution-horizon=8`, `max-chunks=1` executes exactly eight actions and does not
  manufacture an extra chunk; `max-chunks=20` budgets exactly 160 actions.
- [ ] At each handoff the child continues with `B[k:]`, where `k` is measured after
  observation capture and inference. No already-consumed prefix is replayed.
- [ ] With default `--command-conditioning xr`, raw `B` remains finite/in-range and the
  persistent 100 Hz conditioner makes every final outgoing boundary pass the normal
  joint/step checks. With `none`, the raw old-target to `B[k]` check remains hard.
- [ ] A delayed reply that exhausts the overlap causes powered HOLD, not stale
  continuation or fallback to independent chunks.
- [ ] A stale generation is discarded and causes powered HOLD.
- [ ] `s` during inference stops, drains/discards that response, and a second goal repeats
  reset plus the normal warm-start before RTC restarts.
- [ ] `q` and Ctrl-C take the existing orderly release path; they never wait indefinitely
  for a policy reply.
- [ ] Camera timeout, policy timeout, worker failure and action-buffer underrun have each
  been injected in simulation or CPU protocol tests and fail closed.
- [ ] No ZeroMQ camera, policy or DDS socket crosses thread ownership.

Simulation does not prove physical joint signs, real `arm_sdk` authority behavior, DDS
loss behavior, hand stopping, collision safety, or balance safety.

---

## 10. Add adequate run evidence before multi-chunk hardware trials

For every physical experiment, save:

- Console log (`tee`) with monotonic timing, contract result, inference times and cleanup.
- External video showing the entire robot, operator and scene.
- Exact source revisions, checkpoint and dataset hashes.
- Task, initialization mode, execution horizon and maximum chunks.
- Whether release was orderly and acknowledged locally.
- Every anomaly, even if the run appeared successful.

The current console logging is enough for initial hold/one-step observations, but before
multi-chunk trials add a structured per-step trace containing at least timestamps,
commanded arm/hands, measured arm/hands, arm velocity, mode, chunk/step sequence,
inference duration, watchdog/fault state and release outcome.

- [ ] Structured per-step telemetry implemented and replay-checked before multi-chunk
  physical trials.
- [ ] External recording system ready.

---

## 11. Qualify the real robot failure boundary

This is the principal unresolved safety gate. The workstation process cannot guarantee a
bounded release when the Unitree DDS write or physical network itself wedges.

With Unitree-supported procedures, appropriate support/harness, a clear workspace and an
operator on the physical emergency stop, establish and document:

- [ ] Correct robot model and `mode_machine == 6`
  (`g1_29dof_lock_waist_with_hand_rev_1_0`; yaw active, roll/pitch locked),
  matching the data-collection embodiment.
- [ ] Regular/motion mode continues to own the lower body as intended.
- [ ] Exactly one arm/Dex3 DDS command owner exists.
- [ ] `arm_sdk` authority acquisition while holding measured pose causes no jump.
- [ ] Orderly authority ramp-down behaves as expected.
- [ ] Both Dex3 hands respond correctly to Unitree's `stopMotors` command.
- [ ] Defined robot behavior when the arm publisher disappears.
- [ ] Defined Dex3 behavior when each hand publisher disappears.
- [ ] Defined behavior on client crash, network loss, PC power loss and blocked DDS write.
- [ ] A robot-side or independently qualified watchdog/lease handles those failures within
  an accepted time bound.
- [ ] Tracking/rate/freshness thresholds are validated against measured hardware behavior.

Do not improvise network-unplug, process-kill or power-loss experiments on an unsupported
standing robot. Create the failure-injection plan with Unitree guidance and appropriate
physical restraint first.

`STOP`: `--allow-unqualified-real` is only an explicit research override. It does not
satisfy this gate or make a workstation watchdog a robot-side safety system.

---

## 12. First physical stage: measured hold only, no policy action

Preconditions:

- Robot supported and workspace clear.
- Physical emergency-stop operator ready.
- Correct Regular/motion mode.
- Hands empty.
- TeleImager running; XR teleoperation and every other command publisher stopped.
- Server and one-chunk shadow pass immediately beforehand.

Run:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-toothpaste \
    --policy-host 127.0.0.1 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --initialization measured \
    --execution-horizon 1 \
    --max-chunks 1 \
    --actuate \
    --allow-unqualified-real \
    2>&1 | tee real_measured_hold.log
```

Press `r` at the ACTUATE gate. At RUN, press Ctrl-C rather than pressing `r`. This tests
fresh-state qualification, measured hold, authority ramp and orderly release without any
policy action.

- [ ] No visible target jump when authority is acquired.
- [ ] Measured pose remains stable.
- [ ] Ctrl-C reaches the release path.
- [ ] Local log reports a confirmed orderly stop, with no `release_failed` or unconfirmed
  child process.
- [ ] Hands behave as expected after `stopMotors`.

Any motion, unexpected hand behavior or uncertain release is a `STOP`.

---

## 13. Second physical stage: XR-home initialization only

This target sets all 14 arm joints and both seven-joint hand targets to zero. Both hands
must be empty. The path is deliberately slow but is joint-space interpolation, not
collision-aware planning.

Use the same command as the previous gate with:

```bash
--initialization xr-home
```

Press `r` at ACTUATE and then at INITIALIZE. After convergence, visually inspect the
robot. At RUN press Ctrl-C instead of `r`, so no policy action is executed.

- [ ] Initialization begins from the already-published hold target without a jump.
- [ ] Motion is slow, smooth and in the expected direction for every arm/hand joint.
- [ ] No tracking, freshness, mode or convergence fault.
- [ ] Endpoint is stable before the client offers `RUN`.
- [ ] Ctrl-C releases cleanly.

Do not use XR-home for a put-down task while holding an object; it explicitly targets both
hands to zero and can release or disturb the object.

---

## 14. Third physical stage: execute exactly one policy step

XR-home is a neutral staging target, not a demonstrated policy start. The recorded
frame-zero poses are task-dependent and materially different from joint zero. Do not type
`RUN` from XR-home merely because the initialization test passed.

Before the first policy action, establish a task-valid start by one of these routes:

1. Preferred: export and review one real training episode's frame-zero pose, then use
   `--initialization pose-file --initial-pose-file "$INITIAL_POSE_FILE"`.
2. Manually establish and independently verify an approved task-start pose, then use
   `--initialization measured` so the client acquires authority without changing it.

For a pose file:

- [ ] It uses the strict schema documented in `docs/groot_g1_dex3.md`.
- [ ] Task ID and exact instruction match the selected task.
- [ ] It names all 28 joints in exact training order.
- [ ] It contains one reviewed episode's frame-zero arm pose, not a coordinate average.
- [ ] `source` provenance is reviewed manually; the current schema is not a cryptographic
  proof that the numbers came from the declared episode.
- [ ] The complete interpolation from current pose to that target is reviewed for the
  actual scene. Joint limits and rate limits are not collision avoidance.
- [ ] For put-down tasks, object loading and hand policy are reviewed separately. Never
  route through XR-home while holding the object.

If there is no reviewed route into a task-valid pose, this gate is a `STOP`.

### Execute one step

After re-running shadow from the exact scene/start pose:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --initialization pose-file \
    --initial-pose-file "$INITIAL_POSE_FILE" \
    --execution-horizon 1 \
    --max-chunks 1 \
    --actuate \
    --allow-unqualified-real \
    2>&1 | tee real_policy_h1_c1.log
```

Press `r` at ACTUATE, INITIALIZE, and then RUN only after each displayed inspection. If
the approved starting pose was established manually, substitute `--initialization
measured`, omit the pose-file flag, and there will be no INITIALIZE gate.

- [ ] First policy target direction agrees with the shadow prediction and intended task.
- [ ] No discontinuity, limit clipping or unexpected uncommanded joint movement.
- [ ] Lower body remains under the intended Unitree controller.
- [ ] One step completes and cleanup releases cleanly.
- [ ] Measured-versus-commanded tracking is reviewed before proceeding.

---

## 15. Qualify normal ending before object contact

Reaching `max-chunks`, pressing Ctrl-C, or detecting a fault enters cleanup: arm authority
is ramped down and Dex3 `stopMotors` is attempted. This is not a task-level hold state. The
subsequent physical behavior depends on Unitree's controller and hand firmware, and could
allow an object to move or fall.

Before contact with a valuable object:

- [ ] End normally via `max-chunks` at a safe empty-hand pose and record arm/hand behavior.
- [ ] End via Ctrl-C at a safe empty-hand pose and record arm/hand behavior.
- [ ] Establish whether motion mode retakes or changes the arms after authority release.
- [ ] Establish whether each hand holds position, relaxes, or otherwise responds after
  `stopMotors`.
- [ ] If testing a grasp later, begin with a light sacrificial object over a catch area.
- [ ] Define an operator procedure for preserving or safely releasing a held object.

`STOP`: Do not assume that policy completion, `max-chunks`, cleanup, or `stopMotors`
preserves the final grasp/pose.

---

## 16. Expand the physical envelope one variable at a time

Use a new process/run record for each stage:

| Stage | Execution horizon | Max chunks | Approximate commanded action time |
|---|---:|---:|---:|
| A | 1 | 1 | 0.033 s |
| B | 2 | 1 | 0.067 s |
| C | 4 | 1 | 0.133 s |
| D | 8 | 1 | 0.267 s |
| E | 8 | 2 | 0.533 s |
| F | 8 | 4 | 1.067 s |

In synchronous mode wall-clock time is longer because inference occurs between chunks.
Do not increase horizon, chunk count, initialization mode, task and scene difficulty in
the same experiment. Keep this first physical ladder synchronous; RTC needs its own
qualification after the baseline is understood.

For each stage:

- [ ] Re-run a matching publisher-free shadow inference first.
- [ ] Record external video and structured telemetry.
- [ ] Review every target and tracking error before advancing.
- [ ] Confirm cleanup and hand behavior.
- [ ] Repeat enough times to expose intermittent timing/network faults.

Only after these stages pass should `max-chunks` approach a demonstration's task duration.

---

## 17. Task completion and stopping behavior

Current GR00T N1.7 inference does not return a task-complete token. The runner stops only
when:

- `--max-chunks` is reached;
- the operator presses Ctrl-C;
- a validation/watchdog fault occurs; or
- an exception occurs.

At 30 Hz, horizon 8 represents about 0.267 seconds of commanded action per accounting
chunk; 20 chunks represent exactly 160 actions or about 5.33 seconds of commanded motion.
Synchronous mode adds inference gaps between those prefixes. RTC continues the old action
buffer during inference, so it has no deliberate gaps and treats the same values as one
160-action budget. Neither mode infers that the task is finished.

- [ ] Use finite, conservative `max-chunks` for every physical run.
- [ ] Operator remains ready to stop when success or an unsafe condition is observed.
- [ ] Before unattended/longer deployment, add a separately validated completion mechanism
  such as operator approval, task-specific perception, or a trained success classifier.

Small action magnitude or a stationary prediction is not a reliable completion signal by
itself.

---

## 18. Validate RGBD only after colour deployment passes

RGBD is a separate integration layer, even though it uses the same physical RealSense:

- Atomic same-capture JPEG + aligned uint16 depth packet.
- Strictly increasing capture sequence.
- Fresh TeleImager config response and live RealSense scale.
- Exact metric-to-grayscale mapping from the final dataset contract.
- `ego_view` and `depth_gray_view`, both 480x640x3 `uint8`.

- [ ] Freeze/record the XR TeleImager RGBD protocol revision.
- [ ] Visually inspect live grayscale against recorded dataset frames.
- [ ] Verify corrupt, duplicate, stale and restarted streams fail closed.
- [ ] Run a 100-chunk RGBD shadow soak and latency budget.
- [ ] Do not use the current colour-only IsaacLab camera test as proof of RGBD parity.
- [ ] Repeat the entire measured-hold → initialization-only → one-step physical ladder.

---

## 19. Final go/no-go record

### Software/contract

- [ ] Exact revisions and hashes recorded.
- [ ] Automated tests pass.
- [ ] Final checkpoint and dataset contract pass.
- [ ] Full held-out open-loop evaluation reviewed.

### Transport/timing

- [ ] Correct topology proven in shadow mode.
- [ ] Camera and state freshness proven.
- [ ] 100-chunk soak passes.
- [ ] Inference worst case remains below heartbeat deadline with margin.

### Control

- [ ] Isolated IsaacLab stages and fault tests pass.
- [ ] Real measured hold passes.
- [ ] Initialization-only pass.
- [ ] One-step and one-chunk physical stages pass.

### Safety boundary

- [ ] Publisher-loss and Dex3-stop behavior qualified.
- [ ] Independent robot-side failover/watchdog established.
- [ ] No competing command owner.
- [ ] Support, clear workspace and physical emergency-stop operator present.

### Remaining known limitations

- No GR00T task-completion signal.
- RTC is experimental and separately unqualified for physical use; publisher-free shadow
  validates timing/protocol but not closed-loop smoothness.
- Workstation cleanup cannot guarantee release through a failed/wedged DDS path.
- IsaacLab does not prove real balance, collisions, joint signs or hand-stop behavior.

Decision: **GO / NO-GO**

Date: ____________________

Operator/reviewer: _________________________________________

Evidence directory: ________________________________________

Notes:

```text



```
