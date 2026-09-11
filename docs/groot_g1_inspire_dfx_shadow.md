# G1 Inspire DFX shadow evaluation

The guarded GR00T client supports Inspire DFX as an explicit **read-only shadow**
profile. Dex3 remains the default. The Inspire profile creates subscribers only;
`--actuate` is rejected before DDS initialization or any command publisher can be
created.

Start a shadow run with an Inspire-native checkpoint and dataset contract:

```bash
python -m unitree_lerobot.eval_robot.eval_groot_g1 \
  --end-effector inspire-dfx \
  --task pick-wooden-block \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --image-host 192.168.123.164 \
  --network-interface enp132s0 \
  --no-warmup1 \
  --no-gravity-feedforward \
  --execution-horizon 8 \
  --max-chunks 2
```

The checkpoint must advertise `robot_type=Unitree_G1_Inspire_HeadOnly`, a native
26-element state/action vector, exact `left_arm`, `right_arm`, `left_hand`, and
`right_hand` slices, and the converted dataset's complete
`dataset_contract.end_effector` provenance. In particular, its protocol must be
`dfx`; an Inspire FTP checkpoint is rejected before DDS starts.

## DFX state freshness

The official bridge publishes `unitree_go::msg::dds_::MotorStates_` on
`rt/inspire/state`. Its wire order is the six right-hand motors followed by the
six left-hand motors; the policy order is converted to left then right.

The bridge can continue publishing cached positions when a physical hand read
fails. The reader therefore does not equate a DDS callback with a successful hand
sample. It tracks the six `lost` counters for each side independently:

- the first message establishes counter baselines but is not accepted as fresh;
- unchanged counters on the next message accept that side's position and time;
- an increment freezes that side at its last accepted position while the other
  side may continue updating;
- a counter reset or inconsistent six-counter vector invalidates that side until
  a new baseline and subsequent clean sample arrive;
- ordinary age limits still turn a sustained failure into a stale-state error.

Valid all-zero positions are accepted because DFX defines zero as fully closed.

## Why actuation is blocked

There is not yet a reviewed Inspire motion envelope, Inspire-payload gravity
model, initialization target, tracking threshold, or acknowledged motor-stop
operation. The DFX bridge instead stops applying new setpoints after its command
subscription times out (currently one second), which is not equivalent to a
qualified stop command. Consequently, `--actuate`, `--sim`, Warmup1, and Dex3
gravity feed-forward are fail-closed for this profile. Do not treat
`--allow-unqualified-real` as an override for Inspire DFX.
