import copy

import pytest

from robot_ad.schema import InputError, validate_cycle


def _err(payload, rules):
    with pytest.raises(InputError) as e:
        validate_cycle(payload, rules)
    return e.value.code


def test_valid(valid_payload, rules):
    t, x = validate_cycle(valid_payload, rules)
    assert x.shape == (len(t), 18)


def test_missing_channel(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    del p["signals"]["motor_iq_1"]
    assert _err(p, rules) == "missing_channel"


def test_unexpected_channel(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["signals"]["anomaly"] = p["signals"]["motor_iq_1"]
    assert _err(p, rules) == "unexpected_channel"


def test_length_mismatch(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["signals"]["motor_iq_2"] = p["signals"]["motor_iq_2"][:-1]
    assert _err(p, rules) == "length_mismatch"


def test_non_finite(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["signals"]["motor_iq_2"][5] = float("inf")
    assert _err(p, rules) == "non_finite"


def test_time_not_increasing(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["time_s"][10] = p["time_s"][9]
    assert _err(p, rules) == "time_not_increasing"


def test_unknown_version(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["schema_version"] = "robot-cycle-v9"
    assert _err(p, rules) == "unknown_schema_version"


def test_length_out_of_range(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["time_s"] = p["time_s"][:100]
    p["signals"] = {k: v[:100] for k, v in p["signals"].items()}
    assert _err(p, rules) == "length_out_of_range"


def test_sampling_interval(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["time_s"] = [v * 10 for v in p["time_s"]]  # 0.1 s steps
    assert _err(p, rules) == "sampling_interval_out_of_range"


def test_non_numeric(valid_payload, rules):
    p = copy.deepcopy(valid_payload)
    p["signals"]["motor_iq_3"] = ["a"] * len(p["time_s"])
    assert _err(p, rules) == "invalid_type"
