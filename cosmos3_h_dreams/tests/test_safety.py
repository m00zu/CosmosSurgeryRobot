# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safety guard rules on synthetic controller motion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, main

pytestmark = pytest.mark.ci_cpu

DT_NS = 33_333_333


def _frames(positions, *, grips=None, yaw_rates=None, start_ns=0, side="right"):
    """Frames at 30 Hz; the other instrument stays still."""
    frames = []
    for index, position in enumerate(positions):
        angle = (yaw_rates or 0.0) * index * DT_NS / 1e9
        moving = {
            "position": list(position),
            "orientation": [0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)],
            "grip": 0.0 if grips is None else grips[index],
            "trigger": 0.0,
        }
        still = {
            "position": [0.0, 1.0, -0.3],
            "orientation": [0, 0, 0, 1],
            "grip": 0.0,
            "trigger": 0.0,
        }
        other = "left" if side == "right" else "right"
        frames.append(
            {side: moving, other: still, "client_capture_ns": start_ns + index * DT_NS}
        )
    return frames


def _line(speed_m_s, count):
    return [(speed_m_s * i * DT_NS / 1e9, 1.0, -0.3) for i in range(count)]


def _rules(violations):
    return {(v.rule, v.side) for v in violations}


def test_slow_motion_is_safe_and_commits_state():
    guard = SafetyGuard(SafetyLimits())
    assert guard.check(_frames(_line(0.1, 9))) == []


def test_fast_motion_trips_speed_and_names_instrument():
    guard = SafetyGuard(SafetyLimits())
    violations = guard.check(_frames(_line(0.6, 9), side="left"))
    assert ("speed", "left") in _rules(violations)
    speed = next(v for v in violations if v.rule == "speed")
    assert speed.value == pytest.approx(0.6, rel=0.05)
    assert speed.limit == 0.35 and speed.unit == "m/s"
    assert "Left instrument" in speed.message()
    assert speed.to_json()["message"] == speed.message()


def test_sudden_start_trips_acceleration_not_steady_motion():
    limits = SafetyLimits(max_speed_m_s=10.0)
    guard = SafetyGuard(limits)
    steady = _line(1.0, 9)
    assert "acceleration" not in {v.rule for v in guard.check(_frames(steady))}
    guard = SafetyGuard(limits)
    jump = [(0.0, 1.0, -0.3)] * 5 + [
        (1.0 * i * DT_NS / 1e9, 1.0, -0.3) for i in range(1, 5)
    ]
    assert "acceleration" in {v.rule for v in guard.check(_frames(jump))}


def test_fast_rotation_trips_rotation():
    guard = SafetyGuard(SafetyLimits())
    violations = guard.check(_frames(_line(0.0, 9), yaw_rates=5.0))
    rotation = next(v for v in violations if v.rule == "rotation")
    assert rotation.value == pytest.approx(5.0, rel=0.05)


def test_energy_held_too_long_across_chunks():
    guard = SafetyGuard(SafetyLimits(max_energy_s=0.5))
    first = _frames(_line(0.0, 9), grips=[1.0] * 9)
    assert guard.check(first) == []
    second = _frames(_line(0.0, 9), grips=[1.0] * 9, start_ns=8 * DT_NS)
    assert "energy_duration" in {v.rule for v in guard.check(second)}


def test_blocked_window_does_not_commit_energy():
    # Each window holds energy for ~0.267 s: committed bad + calm would exceed 0.4 s.
    guard = SafetyGuard(SafetyLimits(max_energy_s=0.4, max_speed_m_s=0.25))
    bad = _frames(_line(0.6, 9), grips=[1.0] * 9)
    assert guard.check(bad)
    calm = _frames(_line(0.0, 9), grips=[1.0] * 9, start_ns=8 * DT_NS)
    assert guard.check(calm) == []


def test_energy_dose_counts_repeated_taps():
    limits = SafetyLimits(
        max_energy_s=10.0, max_energy_dose_s=0.3, energy_dose_window_s=15.0
    )
    guard = SafetyGuard(limits)
    grips = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    found = set()
    for chunk in range(4):
        found |= {
            v.rule
            for v in guard.check(
                _frames(_line(0.0, 9), grips=grips, start_ns=chunk * 8 * DT_NS)
            )
        }
    assert "energy_dose" in found
    assert "energy_duration" not in found


def test_energy_sweep_measures_distance_during_one_activation():
    guard = SafetyGuard(
        SafetyLimits(
            max_speed_m_s=10.0, max_acceleration_m_s2=1e6, max_energy_sweep_m=0.03
        )
    )
    violations = guard.check(_frames(_line(0.2, 9), grips=[1.0] * 9))
    sweep = next(v for v in violations if v.rule == "energy_sweep")
    assert sweep.value == pytest.approx(0.2 * 8 * DT_NS / 1e9, rel=0.05)


def test_gap_clears_history_so_reset_is_not_a_jump():
    guard = SafetyGuard(SafetyLimits())
    assert guard.check(_frames(_line(0.0, 9))) == []
    far = [(0.5, 1.0, -0.3)] * 9
    assert guard.check(_frames(far, start_ns=8 * DT_NS + 400_000_000)) == []


def test_gap_ends_the_energy_activation():
    guard = SafetyGuard(SafetyLimits(max_energy_s=0.5))
    first = _frames(_line(0.0, 12), grips=[1.0] * 12)  # ~0.37 s of energy
    assert guard.check(first) == []
    later = _frames(_line(0.0, 6), grips=[1.0] * 6, start_ns=11 * DT_NS + 5_000_000_000)
    # ~0.17 s more; 0.53 s if the activation carried across the gap.
    assert guard.check(later) == []


def test_untracked_controller_is_ignored_like_a_missing_one():
    guard = SafetyGuard(SafetyLimits(max_energy_s=0.1, max_energy_sweep_m=0.01))
    assert guard.check(_frames(_line(0.0, 9))) == []
    window = _frames(_line(0.0, 9), grips=[1.0] * 9, start_ns=8 * DT_NS)
    for index, frame in enumerate(window):
        frame["right"].update(tracked=False, position=[5.0 * (index % 2), 9.0, 2.0])
    assert guard.check(window) == []


def test_resume_keeps_dose_but_clears_activation():
    first = _frames(_line(0.0, 9), grips=[1.0] * 9)
    second = _frames(_line(0.0, 9), grips=[1.0] * 9, start_ns=8 * DT_NS)
    guard = SafetyGuard(SafetyLimits(max_energy_s=0.5, max_energy_dose_s=100.0))
    guard.check(first)
    guard.resume()
    assert guard.check(second) == []
    # Each window adds ~0.267 s of dose, so 0.5 s trips only if resume kept the first.
    guard = SafetyGuard(SafetyLimits(max_energy_s=10.0, max_energy_dose_s=0.5))
    assert guard.check(first) == []
    guard.resume()
    assert "energy_dose" in {v.rule for v in guard.check(second)}


def test_limits_from_mapping_validates():
    assert SafetyLimits.from_mapping({"max_speed_m_s": 1}).max_speed_m_s == 1.0
    assert SafetyLimits.from_mapping(None) == SafetyLimits()
    with pytest.raises(ValueError, match="unknown safety"):
        SafetyLimits.from_mapping({"speed": 1})
    with pytest.raises(ValueError, match="positive"):
        SafetyLimits.from_mapping({"max_speed_m_s": 0})
    with pytest.raises(ValueError, match="positive"):
        SafetyLimits.from_mapping({"max_speed_m_s": True})


def test_calibrate_prints_percentiles(tmp_path, capsys):
    actions = np.zeros((40, 28), dtype=np.float32)
    actions[:, 14] = (
        np.arange(40) * 0.1
    )  # right x: translation-scale 10 → 0.01 m per row → 0.1 m/s at 10 fps
    actions[10:20, 24] = 1.0  # right energy for 1 s
    actions[:, 3:9] = actions[:, 17:23] = [1, 0, 0, 0, 1, 0]  # identity rot6d
    path = tmp_path / "t.npy"
    np.save(path, actions)
    main(["calibrate", str(path)])
    out = capsys.readouterr().out
    assert "max_speed_m_s" in out and "0.1" in out
    assert "max_energy_s" in out


def test_check_returns_window_maxima_alongside_unchanged_violations():
    from cosmos3_h_dreams.safety import RULES, SafetyCheck

    guard = SafetyGuard(SafetyLimits())
    slow = guard.check(_frames(_line(0.1, 9), grips=[1.0] * 9))
    assert isinstance(slow, SafetyCheck) and slow == [] and list(slow.ratios) == list(RULES)
    assert slow.maxima["speed"] == pytest.approx(0.1, rel=0.05)
    assert slow.ratios["speed"] == pytest.approx(0.1 / 0.35, rel=0.05)
    assert slow.ratios["energy_duration"] == pytest.approx(8 * DT_NS / 1e9 / 10.0, abs=1e-3)  # ratios have 3 decimals
    assert slow.ratios["rotation"] == 0.0
    fast = SafetyGuard(SafetyLimits()).check(_frames(_line(0.6, 9), side="left"))
    assert ("speed", "left") in _rules(fast)
    assert fast.ratios["speed"] == pytest.approx(0.6 / 0.35, rel=0.05)
    assert fast.maxima["speed"] == pytest.approx(fast[[v.rule for v in fast].index("speed")].value, rel=1e-3)
