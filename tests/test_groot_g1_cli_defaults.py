"""Publisher-free regression checks for the reviewed live Inspire FTP defaults."""

import pytest

from unitree_lerobot.eval_robot.eval_groot_g1 import build_parser, validate_args
from unitree_lerobot.eval_robot.groot_client import DeploymentError


def test_default_profile_is_live_inspire_ftp_with_return_and_acknowledgements():
    args = build_parser().parse_args([])

    assert args.end_effector == "inspire-ftp"
    assert args.actuate is True
    assert args.allow_unqualified_real is True
    assert args.allow_inspire_ftp_unverified_stop is True
    assert args.return_to_start is True
    # Neither select a goal nor introduce an implicit startup movement.
    assert args.task is None
    assert args.custom_goal is None
    assert args.initialization == "measured"
    assert args.network_interface is None
    assert args.confirm_sim_network_isolated is False


def test_bare_live_defaults_still_fail_closed_without_explicit_robot_nic():
    with pytest.raises(DeploymentError, match="explicit --network-interface"):
        validate_args(build_parser().parse_args([]))


def test_usual_live_command_no_longer_needs_the_five_defaulted_flags():
    args = build_parser().parse_args(
        ["--network-interface", "enp2s0", "--initialization", "xr-home"]
    )

    validate_args(args)
    assert args.task is None  # Keep the existing initial task menu.
    assert args.custom_goal is None


def test_default_return_does_not_silently_select_moving_initialization():
    args = build_parser().parse_args(["--network-interface", "enp2s0"])

    with pytest.raises(DeploymentError, match="explicit fixed --initialization xr-home"):
        validate_args(args)
    assert args.initialization == "measured"


def test_single_shadow_opt_out_also_disables_implicit_return():
    args = build_parser().parse_args(["--no-actuate"])

    assert args.actuate is False
    assert args.return_to_start is False
    validate_args(args)


def test_explicit_return_with_shadow_mode_remains_invalid():
    args = build_parser().parse_args(["--no-actuate", "--return-to-start"])

    with pytest.raises(DeploymentError, match="--return-to-start requires --actuate"):
        validate_args(args)


def test_return_can_be_disabled_without_changing_live_mode_or_initialization():
    args = build_parser().parse_args(
        ["--network-interface", "enp2s0", "--no-return-to-start"]
    )

    assert args.actuate is True
    assert args.return_to_start is False
    assert args.initialization == "measured"
    validate_args(args)


@pytest.mark.parametrize(
    ("flag", "attribute", "error"),
    [
        ("--no-allow-unqualified-real", "allow_unqualified_real", "Real actuation is fail-closed"),
        (
            "--no-allow-inspire-ftp-unverified-stop",
            "allow_inspire_ftp_unverified_stop",
            "requires --allow-inspire-ftp-unverified-stop",
        ),
    ],
)
def test_acknowledgement_opt_out_restores_the_existing_live_gate(flag, attribute, error):
    args = build_parser().parse_args(
        ["--network-interface", "enp2s0", "--initialization", "xr-home", flag]
    )

    assert getattr(args, attribute) is False
    with pytest.raises(DeploymentError, match=error):
        validate_args(args)


@pytest.mark.parametrize("profile", ["dex3", "inspire-dfx"])
def test_other_profiles_do_not_inherit_inspire_ftp_acknowledgements(profile):
    args = build_parser().parse_args(["--end-effector", profile, "--no-actuate"])

    assert args.allow_unqualified_real is False
    assert args.allow_inspire_ftp_unverified_stop is False
    validate_args(args)


@pytest.mark.parametrize("profile", ["dex3", "inspire-dfx"])
def test_other_profiles_still_need_explicit_real_override(profile):
    argv = [
        "--end-effector", profile, "--network-interface", "enp2s0",
        "--initialization", "xr-home",
    ]
    with pytest.raises(DeploymentError, match="Real actuation is fail-closed"):
        validate_args(build_parser().parse_args(argv))

    validate_args(build_parser().parse_args([*argv, "--allow-unqualified-real"]))


def test_explicit_ftp_acknowledgement_is_still_invalid_for_other_profiles():
    args = build_parser().parse_args(
        ["--end-effector", "dex3", "--no-actuate", "--allow-inspire-ftp-unverified-stop"]
    )

    with pytest.raises(DeploymentError, match="valid only with --end-effector inspire-ftp"):
        validate_args(args)


@pytest.mark.parametrize("profile", ["dex3", "inspire-ftp"])
def test_simulation_still_requires_network_isolation_and_no_explicit_interface(profile):
    argv = ["--end-effector", profile, "--sim", "--initialization", "xr-home"]
    with pytest.raises(DeploymentError, match="--confirm-sim-network-isolated"):
        validate_args(build_parser().parse_args(argv))

    isolated = [*argv, "--confirm-sim-network-isolated"]
    validate_args(build_parser().parse_args(isolated))
    with pytest.raises(DeploymentError, match="explicit runner --network-interface"):
        validate_args(build_parser().parse_args([*isolated, "--network-interface", "enp2s0"]))


def test_live_loopback_policy_guard_is_unchanged():
    args = build_parser().parse_args(
        [
            "--network-interface", "enp2s0", "--initialization", "xr-home",
            "--policy-host", "192.0.2.1",
        ]
    )
    with pytest.raises(DeploymentError, match="loopback GR00T server"):
        validate_args(args)


def test_parse_known_args_resolves_defaults_without_consuming_unknown_options():
    args, remaining = build_parser().parse_known_args(["--no-actuate", "--external-option"])

    assert args.actuate is False
    assert args.return_to_start is False
    assert args.allow_inspire_ftp_unverified_stop is True
    assert remaining == ["--external-option"]


def test_reusing_parser_does_not_leak_profile_or_actuation_defaults():
    parser = build_parser()
    shadow_dex3 = parser.parse_args(["--end-effector", "dex3", "--no-actuate"])
    live_ftp = parser.parse_args([])

    assert shadow_dex3.return_to_start is False
    assert shadow_dex3.allow_inspire_ftp_unverified_stop is False
    assert live_ftp.return_to_start is True
    assert live_ftp.allow_inspire_ftp_unverified_stop is True


def test_explicit_boolean_flags_keep_last_option_wins_semantics():
    args = build_parser().parse_args(
        [
            "--no-actuate", "--actuate",
            "--return-to-start", "--no-return-to-start",
            "--no-allow-unqualified-real", "--allow-unqualified-real",
            "--allow-inspire-ftp-unverified-stop", "--no-allow-inspire-ftp-unverified-stop",
        ]
    )

    assert args.actuate is True
    assert args.return_to_start is False
    assert args.allow_unqualified_real is True
    assert args.allow_inspire_ftp_unverified_stop is False
