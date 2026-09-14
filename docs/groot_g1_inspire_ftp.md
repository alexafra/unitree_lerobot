# G1 Inspire RH56E2/FTP policy deployment

The guarded GR00T client supports the Inspire RH56E2/T1 hand transport used by
`xr_teleoperate --ee inspire_ftp`. Select it in this client with
`--end-effector inspire-ftp`. Dex3 remains the default and Inspire DFX remains a
separate option; checkpoints and transports are never selected by shape alone.

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

Both acknowledgement flags and the interactive `r` confirmation are required.
`--allow-inspire-ftp-unverified-stop` records the specific fact that publisher
closure is not a verified hand stop.

Arming samples both hands independently and reseeds the first command from the
final fresh DDS-received state. The first hand target is submitted only after
the first matched-pose arm command succeeds. Each final hand write is limited
to 0.2 in normalized units by both the XR conditioner and the writer. This is a
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

Warmup1 is profile-aware. For either Inspire transport it uses one complete,
real 26D `observation.state` row rather than reusing the 28D Dex3 target or
assembling a per-joint mean/median. The frozen source is converted training
episode 56, frame 0, from
`/home/alex/Development/Datasets/lerobot2/inspire_pick_place_red_cup_08_13/train`
(converted timestamp `0.0`; original processed-raw `episode_0079`, frame 0,
task `pick up the red cup.`).
That row was the whole-frame medoid of the 109 training-episode starts and its
recorded image shows both hands empty. It is still a joint-space target, not a
collision-aware plan, and it does not reproduce the recorded legs, waist,
pelvis height, object placement, or world pose. In particular, it is an
empty-hand pick start and does not recreate the put demonstrations' cup-held
right-hand condition.

With the options shown above, startup is an explicit pose chain: guarded
`xr-home`, guarded Inspire Warmup1, RUN, then (when enabled) a fresh inferred
Warmup2 target followed by CONTINUE and a fresh strict inference. Each moving
stage retains its displayed operator gate. Inspire endpoint completion verifies
the arm target and DDS submission, not physical hand convergence, so the
displayed visual hand checks remain required.

`--return-to-start` replays that same enabled fixed pose chain instead of
shortcutting directly to its last target. With `--initialization xr-home
--warmup1`, Shift+Tab therefore moves to XR-home first and then to the Inspire
Warmup1 target, with a separate pre-motion confirmation for each stage and one
post-chain Inspire hand visual check. The next selected goal then follows
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
Warmup1, Warmup2, and each Return-to-Start stage completion therefore mean the
arm endpoint and stability dwell passed while valid hand DDS callbacks
continued and the hand target was submitted; they do **not** mean the hands
were observed to reach the target. After the startup pose chain and Warmup2,
the existing `r` gates explicitly require visual confirmation of both hands. A
Return-to-Start chain has one additional post-chain `r` gate for the same check.
Press `s` at a post-motion gate if either hand did not reach the displayed
target; press `q` to release authority.

At each displayed gate, `r` advances, `s` stays in or enters powered HOLD, and
`q` starts orderly release. During the subsequent blocking startup,
initialization, Warmup1, Warmup2, or Return-to-Start motion, `s` and `q` both
cancel by starting orderly release: the client cannot service the powered-HOLD
barrier until that blocking transition returns. During active policy motion, `s`
instead discards timed work and enters powered HOLD. These software keys do not
replace the physical emergency stop.
