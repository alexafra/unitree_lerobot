# G1 Inspire RH56E2/FTP policy deployment

The guarded GR00T client supports the Inspire RH56E2/T1 hand transport used by
`xr_teleoperate --ee inspire_ftp`. The launch defaults are now Inspire FTP,
live actuation, both real/FTP acknowledgement options, and Return-to-Start.
Your usual command can omit `--end-effector inspire-ftp`, `--actuate`,
`--allow-unqualified-real`, `--allow-inspire-ftp-unverified-stop`, and
`--return-to-start`. Keep the explicit robot network interface and
`--initialization xr-home` for this live reset workflow. Initial pose selection
itself has not changed. Checkpoints and transports are never selected by shape alone.

Use `--no-actuate` for publisher-free shadow mode; it also disables the implicit
Return-to-Start default. Use `--no-return-to-start` to disable resets in a live
run. The two acknowledgement options have explicit `--no-...` forms; denying
either blocks real FTP actuation. Preflight, interactive `r` confirmations,
feedback checks, and stop limitations remain unchanged. Selecting Dex3 or DFX
does not implicitly enable the FTP acknowledgement or real-operation override.

Omit `--task` to start at the task menu. Press Tab for custom-goal entry, or use
`--custom-goal "stack the three cups."` at launch. Explicit custom mode permits
text outside the checkpoint's advertised instructions without an extra override
flag. It still validates the checkpoint task-contract hash and all robot,
vision, and action contracts. Unseen wording may be outside the training
distribution; accepting text does not guarantee successful execution.

## Select a saved Warmup1 pose

Keep the existing episode_0084 measured pose by omitting the pose selector, or
use `--warmup1-pose default`. Select the two saved pyramid poses by name:

```bash
--warmup1-pose pyramid1
```

Or use `--warmup1-pose pyramid2`. `pyramid1` is episode 11 (the first image);
`pyramid2` is episode 46 (the second image). The complete measured poses are
built into the client: no file path or mounted dataset is needed. Both are
Inspire-only frame-0 training states: 14 arm angles in radians and
12 Inspire hand opening fractions. Existing profile and joint limits still apply.

`--warmup1-pose-file PATH` remains available for a future custom full pose;
it cannot be combined with the named selector. The file's profile, units, joint order,
shape, finite values and existing limits are checked before opening robot
connections. An explicit selector with `--no-warmup1` is rejected rather than silently ignored.

The selected pose is loaded once. Startup Warmup1 and Return-to-Start reuse it;
Warmup2 remains independent and does not replace it. Without this option,
existing defaults and motion speeds are unchanged. `warmup1_pose.json` in the
client log folder records the exact values and provenance; file-based selections
include a content SHA256. Vision recording metadata also carries this selection.

These examples are not collision-checked trajectories. Normal confirmation,
feedback and motion limits still apply. The return is the existing smooth
joint-space transition, not obstacle avoidance: inspect the path and recover
an obstructed configuration before returning.

## Opt-in action and joint debugging

Add `--debug-actions` to record a session under the usual client log directory:

```text
logs/eval_groot_g1_<run>/action_debug/
  manifest.json
  events.jsonl
  summary.json
```

The JSONL contains policy requests without images, full returned action chunks,
validated plans, RTC overlap/reference states and handovers, every scheduled
action target, and approximately 30 Hz samples of joint feedback and commands.
Arm position is radians; arm velocity is radians/second; Inspire hands use
normalized opening fractions. Joint names and units are recorded. DDS-accepted
targets and their timestamps are not acknowledgement of physical execution.
For FTP hands, recorded successful-write targets reflect integer `angle_set`
quantization divided by 1000, in canonical profile order.

Policy response actions are server-decoded physical actions, **not** the
normalized latent used inside flow matching. Comparing the previous overlapping
plan with a replacement can reveal physical-space prefix disagreement. This does
not record each denoising step, gradients, images, tactile data, or every 100 Hz
arm write. Use the vision flags separately if wanted.

All producers use a bounded non-blocking queue and a separate writer process.
Overload drops diagnostic samples; it does not make the controller wait for disk.
Inspect `summary.json` and the terminal summary for queue drops, serialization
errors, or missing tails. The session is retained instead of rolling away after
3000 samples, but crash tails can be lost. Logging is off by default, consumes
disk while enabled, and adds some CPU/IPC overhead; no claim of zero overhead.

## Exact 26D contract

- State/action order: left arm 7, right arm 7, left hand 6, right hand 6.
- Per-hand order: pinky, ring, middle, index, thumb bend, thumb rotation.
- Dataset/policy values are `normalized_open_fraction` in `[0, 1]`: zero is
  fully closed and one is fully open.
- FTP feedback is `angle_act[6] / 1000` on `rt/inspire_hand/state/l` and
  `rt/inspire_hand/state/r`.
- FTP commands use angle-control `mode=1` and
  `angle_set[i] = int(normalized[i] * 1000)` on
  `rt/inspire_hand/ctrl/l` and `rt/inspire_hand/ctrl/r`. This intentionally
  matches teleop's non-negative truncation, including values between codes.
- These 0..1000 values are dimensionless angle codes, not radians and not the
  separate 0..2000 actuator-stroke (`pos_*`) representation.

The policy server must advertise `Unitree_G1_Inspire_HeadOnly`, exact `[26]`
state/action shapes and 7/7/6/6 layouts, and an end-effector provenance object
whose protocol is exactly `ftp`. DFX or Dex3 provenance fails before DDS is
initialized.

## SDK prerequisite

The workstation running this client must be able to import the same vendor
`inspire_sdkpy` package/IDL types used by teleop. The package is not copied or
vendored into this repository. Install or expose the robot's reviewed Inspire
SDK in the Python environment, then perform this import-only check:

```bash
/home/alex/miniconda3/envs/unitree_lerobot/bin/python -c 'from inspire_sdkpy import inspire_dds; from inspire_sdkpy.inspire_hand_defaut import get_inspire_hand_ctrl; print(inspire_dds.inspire_hand_state, inspire_dds.inspire_hand_ctrl, get_inspire_hand_ctrl)'
```

The guarded runner repeats this dependency/type preflight before either of its
DDS initialization paths. Missing or incompatible SDK contents therefore fail
without constructing a subscriber or publisher.

## Publisher-free shadow run

Start the matching GR00T server with the FTP colour-only deployment dataset,
then run:

```bash
cd /home/alex/Development/unitree_lerobot
/home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  -m unitree_lerobot.eval_robot.eval_groot_g1 \
  --no-actuate \
  --end-effector inspire-ftp \
  --task pick-red-cup \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --image-host 192.168.123.164 \
  --network-interface enp132s0 \
  --initialization measured \
  --no-warmup1 \
  --no-warmup2 \
  --no-future-goal-warmup2 \
  --no-return-to-start \
  --inference-mode rtc \
  --execution-horizon 8 \
  --max-chunks 2 \
  --command-conditioning xr
```

Shadow mode reads state/camera data and validates policy outputs. It creates no
command publisher. Run this first on the actual PC2/network/SDK combination.

## Optional policy-reference interpolation

### Optional gradient-guided RTC sampler

Add `--rtc-pure` to select asynchronous RTC with inference-time, gradient-guided
action inpainting. The flag itself selects `--inference-mode rtc` (including if
`synchronous` was also supplied). This changes only sampling: use the same
checkpoint and finetunes; no weights are trained or updated. Omit the flag to
keep the existing GR00T copy/freeze/ramp sampler exactly as before.

Restart the policy server with the matching updated PyTorch code first. Before
creating any robot publisher the client requires the server to advertise
`rtc.pure_inference_guidance=true`; every guided continuation must also acknowledge
`rtc_pure_applied=true`. Missing support is an error, never a silent fallback.
The initial chunk (no previous tail) remains ordinary generation. Before the
action clock starts, a discarded guided continuation measures the extra compute
cost and seeds the delay estimator; the original first chunk remains the one
executed. The same calibration runs in publisher-free shadow mode. Later chunks
use clean-action-estimate guidance with an input-gradient backward pass on each
flow step, capped at guidance weight 5; this increases compute and may increase
the number of old actions executed while waiting. Existing delay estimation,
physical arm re-anchoring, hand units, chunk handover and all actuator guards
remain unchanged. Pure mode uses guidance only: neither intermediate denoising
values nor the decoded physical prefix are replaced with the old actions.
Exact prefix restoration remains exclusive to legacy RTC.

`--rtc-frozen-steps` retains its delay-budget meaning; in pure mode it sets the
full-weight guidance region, not a hard-frozen prefix. Do not combine
`--rtc-pure` with `--rtc-ramp-rate`: the legacy velocity ramp is not used. Check
latency and memory using a publisher-free shadow run before any supervised live
comparison; this opt-in is not evidence of improved performance or safety.

### Policy-reference interpolation modes

`--action-interpolation legacy` is the unchanged default: advance desired
policy targets at 30 Hz, with XR measured-relative arm lead and final command
slew conditioning at the 100 Hz publication loop. This is not time-based linear
interpolation. The separate guarded initialization and warmup paths are unchanged.

`--action-interpolation linear --command-conditioning xr` enables an experimental
client-side comparison: sample between consecutive scheduled 30 Hz policy points
at 100 Hz **before** applying the same XR conditioning and safety checks. It adds
no deliberate extra frame delay, cannot overshoot the two reference endpoints,
and holds the last authorized point rather than looking past the action budget.
An RTC replacement leaves the current interval's copied endpoints untouched and
starts using the replacement on the next existing action boundary. STOP/HOLD and
stale-state/replan freezes bypass sampling; no interpolation continues in a hold.

Linear interpolation does not guarantee continuous velocity, acceleration, or
jerk. The conditioned outgoing command can also differ from the reference curve.
It is not collision-aware planning, does not remove synchronous inference gaps,
and is not qualified for unattended real hardware. No new physical velocity or
acceleration limits are assumed. In particular, Inspire's existing per-write hand
step backstop is unchanged and is not a calibrated hand velocity limit.

The option affects actuator execution, not server RTC generation or the
publisher-free shadow prediction checks. Use offline replay or supervised
simulation where supported before an explicitly approved hardware comparison;
Inspire simulation itself remains unqualified. Existing launch commands retain
legacy behaviour unless the new flag is explicitly supplied.

## Explicitly gated supervised actuation

Stop teleop and every other arm/hand command publisher. Support the robot,
clear both hands and the workspace, and keep an operator on the physical
emergency stop.

```bash
cd /home/alex/Development/unitree_lerobot
/home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  -m unitree_lerobot.eval_robot.eval_groot_g1 \
  --actuate \
  --allow-unqualified-real \
  --allow-inspire-ftp-unverified-stop \
  --end-effector inspire-ftp \
  --task pick-red-cup \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --image-host 192.168.123.164 \
  --network-interface enp132s0 \
  --initialization xr-home \
  --warmup1 \
  --warmup2 \
  --future-goal-warmup2 \
  --return-to-start \
  --gravity-feedforward \
  --inference-mode rtc \
  --execution-horizon 8 \
  --max-chunks 2 \
  --command-conditioning xr
```

Both acknowledgement settings remain required for real FTP actuation, but are
enabled by default for this profile. The interactive `r` confirmation remains required.
`--allow-inspire-ftp-unverified-stop` records the specific fact that publisher
closure is not a verified hand stop.

Arming samples both hands independently and reseeds the first command from the
final fresh DDS-received state. The first hand target is submitted only after
the first matched-pose arm command succeeds. Each final hand write is limited
to 0.30 in normalized units by both the XR conditioner and the writer, relative
to the previous successfully written command (initially the measured seed).
The final writer clamps integer wire codes as well, so rounding cannot exceed
that bound. Candidate steps above 0.30 and at most 0.35 are clamped with a
yellow terminal warning (plain text in file logs); steps above the independent
0.35 fault threshold still stop execution.
Invalid values and transport failures remain hard faults. This is a
locally chosen discontinuity backstop, not a manufacturer speed/acceleration
limit.

FTP left and right writes are separate and therefore non-atomic. Both targets
are validated before either message is sent; successful-DDS-Write history is
then recorded independently. If the left Write returns success and the right
Write fails, cleanup records that as a partial write and does not invent or
refresh a right-hand target. A successful DDS Write means only that the local
middleware call returned `True`; it is not an acknowledgement from the bridge
or physical hand.

`xr-home` uses an Inspire-specific staging pose: shoulders and wrists remain at
joint zero, both elbows target `-0.15 rad` to raise the lower Inspire hands
slightly, and both hands are all one (fully open). The move follows a bounded
interpolation, but it can drop an object. Both hands must be empty.

At initial startup only, an additional guarded settle follows `xr-home` and
changes both elbow targets from `-0.15 rad` to `-0.05 rad` while retaining the
zero shoulder/wrist and fully-open hand targets. This stage runs whether or not
Warmup1 or Warmup2 is enabled. Return-to-Start goes directly to Warmup1 when
enabled, otherwise this -0.05 desk target; it never replays the -0.15 walking pose.

Warmup1 is profile-aware. For either Inspire transport it uses one complete,
real 26D `observation.state` row rather than reusing the 28D Dex3 target or
assembling a per-joint mean/median. The frozen source is converted training
episode 428, frame 0, from
`/home/alex/Development/Datasets/lerobot2/inspire/all_tasks_713eps_20260917_normals_range_mask_v2/train`
(converted timestamp `0.0`; original processed-raw stack `episode_0084`, frame
0, task `stack the three red cups.`). That row is the whole-frame medoid of the
115 stack-task training episode starts. It is still a joint-space target, not a
collision-aware plan, and it does not reproduce the recorded legs, waist,
pelvis height, object placement, or world pose.

Warmup1 now uses `1.68x` the ordinary guarded interpolation rate; Warmup2 uses
`1.26x`. Both are another 20% faster than their previous `1.40x` / `1.05x`
rates (about 16.7% less rate-limited interpolation time). Convergence dwell,
feedback pauses, XR-home and the initial-only elbow-settle rate are unchanged.
Inspire FTP hand commands remain capped by the `0.30` normalized writer clamp,
with an independent `0.35` excessive-step fault threshold.

For both Inspire hand transports, live arm **command** targets also restrict
wrist yaw to protect the external hand wiring: in the G1 zero-pose front view,
left wrist yaw is `[-80°, +60°]` and right wrist yaw is `[-60°, +80°]`.
The 60° side is outward from the torso on each arm. Policy predictions are
clamped to these bounds; guarded moving poses and final commands are checked
against them. Unitree's physical joint limits and measured-state validation
remain unchanged. This does not apply a new margin to other joints, and an
already-running evaluation client must be restarted to load this change.

With the options shown above, startup is an explicit pose chain: guarded
`xr-home`, guarded initial-only elbow settle, guarded Inspire Warmup1, RUN, then
(when enabled) a fresh inferred Warmup2 target followed by CONTINUE and a fresh
strict inference. Each moving stage retains its displayed operator gate.
Inspire endpoint completion verifies the arm target and DDS submission, not
physical hand convergence, so the displayed visual hand checks remain required.

`--return-to-start` goes directly to the reviewed Warmup1 pose when enabled,
without an intermediate desk/home pose. With `--no-warmup1`, it goes directly
to the desk pose (both elbows **-0.05 rad**, other arm joints zero, hands open).
The **-0.15 rad** XR-home walking/staging pose is used only at initial startup;
it is never replayed by Return-to-Start. With `--no-warmup1`, the reset ends at
the -0.05 desk pose. Shift+Tab requests this reset from powered HOLD, with a
separate pre-motion confirmation and one final Inspire hand visual check.
Return motion keeps its additional **1.596x** stage multiplier: Warmup1 return
therefore uses `1.596 * 1.68 = 2.68128x` the ordinary rate, while a no-Warmup1
desk return remains `1.596x`, subject to existing hard limits. This return-only
multiplier does not apply to startup or policy execution. Warmup2 never changes
the reset target. The next selected goal still follows
`--warmup2` when enabled. Inspire Return-to-Start deliberately requires the
fixed `xr-home` initialization even when Warmup1 is enabled; measured
initialization is not accepted for this workflow. Return-to-Start never
reacquires command authority and never reuses an old policy result.

## Stop and feedback limitations

The two hand streams have independent receipt timestamps; a stale side pauses
motion even if the other continues. Recovery requires newer paired samples and
the existing bounded recovery gate. The client validates exact six-value
`angle_act` shape, finiteness and the 0..1000 wire range before normalizing.

The FTP state IDL contains no device-read timestamp, sequence number, or lost
counter. Consequently, “fresh” here means a recent valid DDS callback. The
client cannot distinguish a genuinely new physical hand read from a bridge
repeatedly publishing a cached valid `angle_act` sample. Unchanged values also
cannot be rejected because a stationary hand legitimately repeats them. Verify
the reviewed bridge is healthy before actuation; DDS receipt freshness alone is
not proof of a new device read, command execution, or hand convergence.

No FTP command-expiry behavior, motor-stop command, or stop acknowledgement has
been qualified in this client. On `q`, Ctrl-C, normal completion, or a fault,
the client ramps arm authority to zero first and then closes both FTP command
publishers without sending a cleanup hand target. Closing publishers must not
be interpreted as a hand stop; conservatively treat any hand setpoint from a
successful DDS Write as potentially still active. The run log records whether
neither, one, or both per-side DDS Writes ever completed; it does not report
bridge or device acceptance. The physical emergency stop remains the
authoritative stop mechanism.

There is no qualified Inspire hand tracking-error threshold. Initialization,
startup elbow settle, Warmup1, Warmup2, and each Return-to-Start stage
completion therefore mean the
arm endpoint and stability dwell passed while valid hand DDS callbacks
continued and the hand target was submitted; they do **not** mean the hands
were observed to reach the target. After the startup pose chain and Warmup2,
the existing `r` gates explicitly require visual confirmation of both hands. A
Return-to-Start chain has one additional post-chain `r` gate for the same check.
Press `s` at a post-motion gate if either hand did not reach the displayed
target; press `q` to release authority.

At each displayed gate, `r` advances, `s` stays in or enters powered HOLD, and
`q` starts orderly release. During the subsequent blocking startup,
initialization, startup elbow settle, Warmup1, Warmup2, or Return-to-Start
motion, `s` and `q` both
cancel by starting orderly release: the client cannot service the powered-HOLD
barrier until that blocking transition returns. During active policy motion, `s`
instead discards timed work and enters powered HOLD. These software keys do not
replace the physical emergency stop.

## Continuous demo footage and preview

Add `--record-full --show-camera` to the existing client command. The preview
and full recording share a separate, low-priority camera subscriber process,
independent of policy requests. They target 30 FPS (or the camera's slower
configured rate); `--demo-fps 15` lowers that target. This does not change
inference frequency, action timing, conditioning, or robot commands.

The demo starts before preflight/initialization and continues across warmups,
powered holds and goal changes until client exit. Full recording lives at
`Recordings_Data/<short-model-label>/<checkpoint>/YYYY-MM-DD_HH-MM/full_demo/`.
Labels include `rgb_final_870ep` and `turbo_latepre_final_870ep`; full model names
remain in each manifest. There are no seconds or microseconds in the folder
timestamp. Repeat runs in the same minute use a `run02_` prefix, never overwrite.
RGB and checkpoint-selected
Turbo depth or normals are lossless PNGs. `frames.jsonl` preserves actual sample
and source-receipt timestamps; `events.jsonl` contains phase markers. Show a
shared visible timer at the start and end to align external phone footage.
Do not use a fixed 5-FPS export for synchronization: use the saved timestamps.

`--record-vision` remains independent: it records policy captures (including
discarded preflight/safety recaptures), not the continuous demo feed. Both flags
can be used together; policy frames remain in the named run folder and continuous
frames in its `full_demo` child. Model/checkpoint names require the updated
policy server; restart it only once the robot is safely stopped.

The new process adds camera bandwidth and CPU/disk load; actual FPS is not
guaranteed. Duplicate cached frames are skipped and overloaded recording queues
drop frames rather than blocking control. Check the demo summary for drops and
errors. Closing the demo window or pressing `q` requests normal client release,
not a guaranteed hand stop. Record/preview failures after startup do not command
the robot. No full-mode hardware throughput has been qualified by the offline tests.

## Change the model during powered HOLD

The server accepts `--model rgb`, `--model separate`, or `--model fusion`.
`separate` selects the surface-normal separate-view checkpoint; `fusion`
selects corrected Turbo depth, pre-adapter late fusion. All three aliases
resolve to pinned 30k checkpoints under
`/home/alex/Development/Models/inspire_live_test/live_models.json`.
The existing `--model-path` remains available for manual experiments.

For blinded trials, start the server with `--blind A` or `--blind B` and
`--blind-study live-ab-01`. They select only RGB and corrected Turbo late
fusion; the surface-normal separate-view checkpoint is excluded. The two
letters retain their checkpoint assignments across server restarts. The study mapping is private under the
live-model folder and the readable key is
`/home/alex/Desktop/inspire_AB_key.txt`. Do not inspect either during scoring.
Normal terminal output and client recording folders use only the letter;
the live preview shows RGB only. Recorded geometry and private diagnostics
remain available for later analysis, so this is operator blinding rather than
protection against someone intentionally opening the raw files.

The server command, from `/home/alex/Development/Isaac-GR00T`, is:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m gr00t.eval.run_gr00t_server \
  --model separate --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 --host 127.0.0.1 --port 5555
```

For a blind trial, replace `--model separate` with
`--blind A --blind-study live-ab-01`. In either case the server uses the pinned
deployment dataset contract from the final train view unless explicitly
overridden.

While the **client is in powered HOLD**, stop the old policy server and start
the new one. Press **Ctrl+R** in the client's HOLD menu (or type `refresh` and
Enter). The client reconnects, validates the new model and camera encoding,
closes the old camera and recording segment, starts a new segment, and
discards a fresh preflight action. It then remains in HOLD until a goal is
explicitly selected. The actuator and state reader stay running. If refresh
fails, stay in HOLD and retry Ctrl+R after fixing the server; `q` still releases
authority. The new code must already be in the running client, so restart the
client once before the first use of this feature.
