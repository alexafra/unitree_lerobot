# Laptop GR00T shadow-mode checklist

This checklist is for testing the real G1 camera/state-to-GR00T pipeline from the XR
laptop **without sending any robot commands**.

The intended topology is:

```text
G1 PC2: TeleImager camera server
    |
    | robot Ethernet (192.168.123.x)
    v
XR laptop: Unitree GR00T client, camera subscriber, DDS state subscriber
    |
    | Wi-Fi, encrypted SSH tunnel
    v
GPU PC: Isaac-GR00T PolicyServer on 127.0.0.1:5555
```

The laptop client must remain in its default shadow mode. Every client command below
deliberately omits `--actuate`, `--allow-unqualified-real`, and `--sim`. Shadow mode
does not construct arm or Dex3 command publishers.

## 1. Freeze and record the software versions

On the GPU PC:

```bash
cd "$HOME/Development/Isaac-GR00T"
git rev-parse HEAD
git status --short
```

On the laptop:

```bash
cd "$HOME/Development/unitree_lerobot"
git rev-parse HEAD
git status --short

conda activate unitree_lerobot
python -m pip install -e .
python -m unitree_lerobot.eval_robot.eval_groot_g1 --help
```

Record both commit IDs and any intentional dirty diff with the test log. The laptop
must contain the current GR00T client/contract code, not an older checkout.

## 2. Start the camera server on G1 PC2

In PC2's prepared TeleImager environment:

```bash
python -m teleimager.image_server --rs
```

Keep only the camera server running. Do not run `teleop_hand_and_arm.py`, a replay
program, `eval_g1.py`, or another arm/hand controller during this test.

For the first laptop test, use the colour-only checkpoint. RGBD requires the updated
atomic TeleImager RGBD server/client protocol and should be tested separately.

## 3. Start the model server on the GPU PC

The existing colour checkpoint can be used to test the network before the final
retrain:

```bash
cd "$HOME/Development/Isaac-GR00T"

DEPLOY_MODEL="$HOME/Development/Models/combined_colour_only_batch_32_acc_1_0908_2/checkpoint-10000"
DEPLOY_DATASET="$HOME/Development/Datasets/lerobot2/combined_atomic_only_09_08_2/train"

CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m gr00t.eval.run_gr00t_server \
    --model-path "$DEPLOY_MODEL" \
    --embodiment-tag NEW_EMBODIMENT \
    --deployment-dataset-path "$DEPLOY_DATASET" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port 5555
```

Keep the server bound to `127.0.0.1`. The raw GR00T ZeroMQ endpoint is not an
authenticated Wi-Fi service. The SSH tunnel in the next step makes this loopback-only
server appear at `127.0.0.1:5555` on the laptop.

After retraining, replace `DEPLOY_MODEL` and `DEPLOY_DATASET` together. The dataset
path must be the exact training dataset recorded by that checkpoint; it is used to
export and verify the robot, joint-order, frame-rate, video, and action-semantics
contract. It is not loaded for inference samples.

## 4. Identify the laptop's two network routes

On the laptop:

```bash
ip -br -4 addr
ip route get 192.168.123.164
```

In the second command's output, the name after `dev` is the robot Ethernet interface.
For example, if it says `dev enp3s0`, use:

```bash
ROBOT_NIC="enp3s0"
```

Do not copy that example name blindly. Confirm that PC2 is reached through the chosen
interface:

```bash
ping -I "$ROBOT_NIC" -c 3 192.168.123.164
```

Find the GPU PC's Wi-Fi address by running `ip -br -4 addr` on the GPU PC. Then, on
the laptop, verify that the route to it is the Wi-Fi interface:

```bash
GPU_PC_WIFI_IP="REPLACE_WITH_GPU_PC_WIFI_IP"
ip route get "$GPU_PC_WIFI_IP"
ping -c 20 "$GPU_PC_WIFI_IP"
```

Do not proceed until PC2 routes over robot Ethernet and the GPU PC routes over Wi-Fi.

## 5. Create the SSH tunnel from the laptop

In a dedicated laptop terminal:

```bash
GPU_PC_WIFI_IP="REPLACE_WITH_GPU_PC_WIFI_IP"

ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=3 \
    -L 127.0.0.1:5555:127.0.0.1:5555 \
    "alex@$GPU_PC_WIFI_IP"
```

Leave that terminal open. The two uses of port 5555 do not conflict: the GR00T client
connects to laptop loopback, while the camera client connects to PC2 at
`192.168.123.164`.

In another laptop terminal, confirm that SSH owns the local listener:

```bash
ss -ltnp | rg '127\.0\.0\.1:5555'
```

## 6. Run a short visual shadow test on the laptop

Use one exact trained task first, so language generalization is not mixed into the
network test:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot

ROBOT_NIC="REPLACE_WITH_EXPLICIT_ROBOT_ETHERNET_INTERFACE"

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --policy-port 5555 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --initialization measured \
    --execution-horizon 8 \
    --max-chunks 5 \
    --show-camera
```

Required output includes:

```text
GR00T contract verified
TeleImager head stream is live
Publisher-free preflight passed
SHADOW MODE: no command publishers were created
Shadow chunk 5/5
```

The preview is the decoded `ego_view` actually sent to GR00T. Pressing `q` in that
window ends the shadow runner. It does not mean the terminal `q` command used by an
armed multi-goal session because this run is never armed.

## 7. Run and save a latency sample

Omit the preview while measuring latency:

```bash
cd "$HOME/Development/unitree_lerobot"
conda activate unitree_lerobot

ROBOT_NIC="REPLACE_WITH_EXPLICIT_ROBOT_ETHERNET_INTERFACE"
SHADOW_LOG="laptop_shadow_h8_c100.log"

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --policy-port 5555 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --initialization measured \
    --execution-horizon 8 \
    --max-chunks 100 \
    2>&1 | tee "$SHADOW_LOG"
```

Extract request latency and observed replan rate:

```bash
python - "$SHADOW_LOG" <<'PY'
from datetime import datetime
from pathlib import Path
import math
import re
import statistics
import sys

pattern = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"Shadow chunk \d+/\d+: inference ([0-9.]+)s"
)
samples = []
times = []
for line in Path(sys.argv[1]).read_text().splitlines():
    match = pattern.search(line)
    if match:
        times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f"))
        samples.append(float(match.group(2)))

if not samples:
    raise SystemExit("No shadow-chunk inference samples found")

ordered = sorted(samples)
def percentile(p):
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

print(f"samples: {len(samples)}")
print(f"mean:    {statistics.fmean(samples):.3f} s")
print(f"median:  {statistics.median(samples):.3f} s")
print(f"p95:     {percentile(0.95):.3f} s")
print(f"p99:     {percentile(0.99):.3f} s")
print(f"max:     {max(samples):.3f} s")
if len(times) > 1:
    elapsed = (times[-1] - times[0]).total_seconds()
    print(f"replans: {(len(times) - 1) / elapsed:.2f} Hz over {elapsed:.1f} s")
PY
```

The logged `inference` duration is end-to-end from the laptop's policy request until
the response returns. It includes serialization, transport of roughly 0.88 MiB of raw
480x640 RGB through SSH/Wi-Fi, server preprocessing, model denoising, action
postprocessing, and the reply. It does not include the camera/state capture performed
immediately before the timer starts.

For reference, a captured same-PC run was approximately 0.142 s mean, 0.188 s p95,
and 0.198 s maximum. Compare the laptop p50/p95/p99/max and timeout count with that
baseline. The ping result measures small-packet network latency only; the GR00T request
timing is the more useful test of image-transfer performance.

`--execution-horizon 8` does not make one inference request eight times larger or make
the model predict only eight actions. The checkpoint returns its full configured chunk,
the client validates the full response, and synchronous shadow evaluates the first eight
before requesting another chunk. It intentionally waits `8/30 = 0.267 s` between shadow
requests, so the replan interval includes that wait plus inference time.

## 8. Optional RTC wire/timing shadow

After the synchronous network baseline passes, a checkpoint exposing at least 32 actions
can exercise the RTC v1 request protocol without publishers:

```bash
RTC_SHADOW_LOG="laptop_rtc_shadow_h8_c100.log"

python -m unitree_lerobot.eval_robot.eval_groot_g1 \
    --task pick-red-cup \
    --policy-host 127.0.0.1 \
    --policy-port 5555 \
    --image-host 192.168.123.164 \
    --network-interface "$ROBOT_NIC" \
    --inference-mode rtc \
    --execution-horizon 8 \
    --max-chunks 100 \
    2>&1 | tee "$RTC_SHADOW_LOG"
```

RTC shadow uses a virtual 30 Hz action clock. For each request it captures the
pre-capture virtual plan index, sends that still-unconsumed physical tail, and measures
how many virtual actions elapse through camera capture, Wi-Fi/SSH, inference, parsing and
handoff. It sends no commands and the physical robot state does not follow the virtual
plan, so this tests capability negotiation, serialization, latency budget, stale reply
handling and buffer underrun—not RTC motion smoothness or task quality.

Leave `--rtc-frozen-steps` unset for this first measurement so the client estimates the
post-capture delay and adds each request's observed capture cost once. Save the request
index, overlap, frozen prefix, inference time, actual elapsed actions and handoff logs.
An overlap exhaustion is a useful failed timing result; it must never silently become an
independent asynchronous chunk.

## 9. Shadow pass/fail checklist

- [ ] GPU server remains bound to `127.0.0.1:5555`.
- [ ] Laptop SSH tunnel is stable for the entire test.
- [ ] Laptop reaches PC2 through the explicit Ethernet interface.
- [ ] Contract reports the expected video keys, 28-joint ordering, 30 Hz, and model
      horizon.
- [ ] TeleImager reports fresh 480x640 frames at approximately 30 FPS.
- [ ] Output explicitly says `SHADOW MODE: no command publishers were created`.
- [ ] The preview matches the intended robot view.
- [ ] The 5-chunk visual test and 100-chunk latency test complete without a policy,
      camera, DDS-state, stale-frame, or contract timeout.
- [ ] Server, laptop-client, and PC2-camera logs are saved with commit IDs.
- [ ] Disconnecting the SSH tunnel causes a bounded client error and still creates no
      command publisher.

An action jump or joint-limit rejection means the network path worked and the returned
policy output failed a contract check. Do not relax motion limits merely to complete a
network-only shadow test.

This checklist does not authorize real movement. Shadow validates observation assembly,
transport, inference, model contract, and action validation. It does not validate robot
command mapping, gains, collision safety, publisher-loss behavior, or physical release.
See [the full validation checklist](groot_g1_pipeline_validation_checklist.md) before any
real actuation.

## 10. Paste-ready summary for Codex on the laptop

Paste the following into a Codex session running in the laptop's
`~/Development/unitree_lerobot` checkout:

```text
I am validating the Unitree G1-29 + Dex3 GR00T pipeline in publisher-free shadow mode
from this laptop. Do not add --actuate, --allow-unqualified-real, --sim, or create any
DDS command publisher.

Topology:
- G1 PC2 at 192.168.123.164 runs only `python -m teleimager.image_server --rs`.
- This laptop reaches PC2 and robot DDS over its wired robot-Ethernet interface.
- Isaac-GR00T PolicyServer runs on the GPU PC at its own 127.0.0.1:5555.
- This laptop reaches that server over Wi-Fi through SSH local forwarding:
  laptop 127.0.0.1:5555 -> GPU PC 127.0.0.1:5555.
- XR teleoperation, eval/replay tools, and all other arm/Dex3 command writers are off.

Use docs/groot_g1_laptop_shadow.md as the runbook. Help me:
1. inspect `ip -br -4 addr` and `ip route get 192.168.123.164` to identify the explicit
   robot Ethernet interface without guessing;
2. verify the GPU-PC route is Wi-Fi and the SSH tunnel owns laptop 127.0.0.1:5555;
3. run the exact trained-task 5-chunk visual shadow test;
4. run the 100-chunk, horizon-8 shadow latency test and preserve its log;
5. calculate mean/median/p95/p99/max request latency and observed replan rate;
6. compare it with the same-PC reference (mean ~0.142 s, p95 ~0.188 s, max ~0.198 s);
7. only after synchronous shadow passes, optionally run the documented 32+-action RTC
   virtual-clock shadow and preserve request-index/overlap/frozen/delay logs;
8. stop on any contract/camera/state/action error and diagnose it without weakening
   deployment safety constants.

The expected client output must explicitly include:
`SHADOW MODE: no command publishers were created`.
```
