# G1 Inspire DFX policy deployment

The guarded GR00T client supports Inspire DFX in publisher-free shadow mode and
in explicitly authorized, supervised real actuation. Dex3 remains the default
and uses its existing 28-dimensional contract. Inspire checkpoints use a native
26-dimensional contract and cannot be mixed with Dex3 data or models.

## Exact contract

- Model/state order: left arm 7, right arm 7, left hand 6, right hand 6.
- Per-hand order: pinky, ring, middle, index, thumb bend, thumb rotation.
- Hand unit: `normalized_open_fraction` in `[0, 1]`; zero is closed and one is open.
- DFX DDS wire order is right hand at indices 0..5, then left hand at 6..11 on
  `rt/inspire/state` and `rt/inspire/cmd`.
- One combined `MotorCmds_` DDS sample carries both hand targets together.

The policy server must advertise `robot_type=Unitree_G1_Inspire_HeadOnly`, native
`[26]` state/action shapes, exact 7/7/6/6 partitions and names, and complete DFX
end-effector provenance. FTP or Dex3 metadata is rejected before publishers are
created.

## Shadow first

Shadow mode subscribes to real robot and camera state but creates no command
publisher:

```bash
cd /home/alex/Development/unitree_lerobot
/home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  -m unitree_lerobot.eval_robot.eval_groot_g1 \
  --end-effector inspire-dfx \
  --task stack-three-cups \
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

## Supervised live command

Stop XR teleoperation and every other arm/hand publisher first. Restrain and
support the robot, clear the workspace, keep the terminal focused, and have an
operator on the physical emergency stop.

```bash
cd /home/alex/Development/unitree_lerobot
/home/alex/miniconda3/envs/unitree_lerobot/bin/python \
  -m unitree_lerobot.eval_robot.eval_groot_g1 \
  --actuate \
  --allow-unqualified-real \
  --end-effector inspire-dfx \
  --task stack-three-cups \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --image-host 192.168.123.164 \
  --network-interface enp132s0 \
  --initialization xr-home \
  --no-warmup1 \
  --warmup2 \
  --future-goal-warmup2 \
  --return-to-start \
  --gravity-feedforward \
  --inference-mode rtc \
  --execution-horizon 8 \
  --max-chunks 2 \
  --command-conditioning xr
```

`--allow-unqualified-real` is mandatory because the choices below are
laboratory choices, not manufacturer-qualified limits or safety certification.

### What arming and initialization do

1. Before command creation, the client validates the checkpoint, dataset,
   camera, 26D layout, DFX provenance, and one publisher-free inference.
2. At the ACTUATE prompt, `r` creates the guarded child and publishers; `s` or
   `q` cancels while no authority exists. Construction sends no hand packet.
3. The child requires fresh arm/hand feedback and a stationary measured dwell.
   The latest measured finger q becomes the first DFX target.
4. The first command is measured arm q at full `arm_sdk` authority with zero
   feed-forward torque. Only after it succeeds is the measured combined-hand
   hold sent.
5. Arm q stays fixed while gravity torque ramps in using the same
   `g1_body29_hand14.urdf` used by Inspire teleop. The unchanged hand command is
   refreshed throughout ramp and settle so DFX's roughly one-second lease does
   not expire.
6. The command above explicitly selects XR-home, preserving the original Inspire
   teleop startup pose: all 14 arm targets are zero and all six normalized
   channels of each hand are one (fully open). The guarded client interpolates
   to it instead of issuing a discontinuous startup command. It can drop an
   object, so both hands must be empty. `--initialization measured` remains the
   default no-home-motion alternative.
7. Warmup1 is disabled because there is no reviewed 26D training-frame home. At
   INITIALIZE, RUN, and WARMUP2, `r` advances after inspection. Warmup2 reaches
   only target zero of a fresh policy chunk, discards it, then resets and
   re-observes for ordinary execution.

The gravity model provides teleop parity, not an Inspire-specific payload
identification. `--no-gravity-feedforward` remains available for a separately
supervised zero-torque comparison.

### The 0.2 normalized command bound

Raw 30 Hz policy targets may span `[0, 1]`. The 100 Hz XR conditioner limits
each final finger command to at most `0.2` normalized units from the last
command. The writer independently applies the same check against the last
command DDS accepted, and validates both hands before mutating or sending the
combined message.

This is a per-write discontinuity backstop, not a speed or acceleration rating:
repeated 100 Hz writes can traverse more than 0.2 per second. It is derived from
the existing teleop behavior and remains explicitly unqualified.

### Feedback loss and HOLD

DFX may publish cached q when a physical hand read fails, so callback freshness
alone is insufficient. The state reader tracks each side's six
`MotorState.lost` counters. An increment freezes only that side; a reset or
divergent counter vector invalidates it. A malformed combined message is ignored
and cannot refresh either side; ordinary age checks then make stale data fail.

A lost-counter change or state age over 100 ms immediately freezes the last
outgoing targets and discards active and in-flight timed work. Three genuinely
newer paired clean samples are required before re-observation and replanning. If
the pause lasts 1.25 seconds, automatic resume is revoked and the client enters
operator HOLD. A sustained hard-age violation faults the actuator. Discarded
actions are never replayed.

There is no manufacturer-qualified normalized hand tracking-error threshold in
this profile. Hand position error is recorded as a diagnostic, but it does not
itself trigger a warning or cutoff. The hard hand-side gates are lost counters,
freshness, valid `[0, 1]` range, the conditioned/write-step bound, heartbeat,
and DDS write success. Arm tracking remains hard-gated by the existing client.

### `r`, `s`, and `q`

- `r` advances each displayed authority or motion gate; it never resumes an
  abandoned timed plan.
- During policy motion, `s` discards timed work and enters a powered 100 Hz
  position HOLD at the last commanded arm/hand targets. It is not a passive
  brake and can continue exerting force. A newly selected goal is freshly
  inferred; initialization and Warmup1 are not repeated.
- During guarded initialization/Warmup2 interpolation, `q` is the immediate
  orderly-abort key; `s` is deliberately not a mid-interpolation pause.
- During active motion or HOLD, `q` starts orderly release and exit. `Ctrl-C`
  uses the same cleanup path.

For Inspire DFX, Return-to-Start is offered only when the run explicitly selects
`--initialization xr-home --return-to-start`. Shift+Tab from powered HOLD asks
for a fresh `r` confirmation, then follows the same bounded path back to zero
arms and fully open hands. It does not repeat authority acquisition or Warmup1;
the next goal follows the ordinary Warmup2 gate. Return-to-Start remains rejected
with measured initialization because that mode deliberately has no fixed target.

### Lease-only hand release

On `q`, Ctrl-C, a watchdog fault, or normal completion, the client first ramps
`arm_sdk` authority to zero while refreshing only the last successfully
accepted DFX hand hold. A newer target whose hand write failed is never promoted
during cleanup. If no hand packet ever succeeded, cleanup does not acquire a
new hand lease. After the final zero-authority arm write, the publisher closes
and sends no further hand packet.

DFX exposes no hand motor-stop command or acknowledgement here. The bridge
therefore stops receiving SetPosition refreshes and is expected to let its
roughly one-second lease expire. The client logs lease relinquishment, never an
acknowledged motor stop. DDS/network failure, process kill, host power loss, or
a robot-side fault can bypass or delay software cleanup; the physical emergency
stop remains authoritative.
