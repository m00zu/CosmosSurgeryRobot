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

"""Which tools each instrument has, from the session config's ``instruments`` block.

```yaml
instruments:
  left: {name: Grasper, tools: [jaws]}
  right: {name: Hook, tools: [energy]}
```

Without the block both instruments have both tools. The server drops a control
an instrument lacks (trigger for jaws, grip for energy) before it reaches the
model, the safety guard or a recording, and course files may not ask for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SIDES = ("left", "right")
TOOLS = ("jaws", "energy")
_CONTROL = {"jaws": "trigger", "energy": "grip"}
"""The xr_frame controller field that drives each tool."""
_NAME_MAX = 30


@dataclass(frozen=True)
class Instrument:
    name: str
    tools: frozenset[str]

    def label(self, side: str) -> str:
        """``"right instrument (Hook)"`` for error messages."""
        return f"{side} instrument ({self.name})"


DEFAULT_INSTRUMENTS = {
    side: Instrument(name=side.capitalize(), tools=frozenset(TOOLS)) for side in SIDES
}


def parse_instruments(raw: object) -> dict[str, Instrument]:
    """Validate the ``instruments`` block; ``None`` gives both tools on both sides.

    Raises:
        ValueError: ``"instruments.<side>.<field>: <problem>"``.
    """
    if raw is None:
        return dict(DEFAULT_INSTRUMENTS)
    if not isinstance(raw, Mapping) or set(raw) != set(SIDES):
        raise ValueError("instruments: must have exactly the keys left and right")
    instruments = {}
    for side in SIDES:
        entry = raw[side]
        if not isinstance(entry, Mapping) or set(entry) - {"name", "tools"}:
            raise ValueError(f"instruments.{side}: allowed keys are name and tools")
        name = entry.get("name", side.capitalize())
        if not isinstance(name, str) or not name.strip() or len(name) > _NAME_MAX:
            raise ValueError(
                f"instruments.{side}.name: must be text of at most {_NAME_MAX} characters"
            )
        tools = entry.get("tools")
        if (
            not isinstance(tools, list)
            or any(tool not in TOOLS for tool in tools)
            or len(set(tools)) != len(tools)
        ):
            raise ValueError(
                f"instruments.{side}.tools: must be a list of {' and/or '.join(TOOLS)}"
            )
        instruments[side] = Instrument(name=name.strip(), tools=frozenset(tools))
    return instruments


def instruments_json(instruments: Mapping[str, Instrument]) -> dict[str, Any]:
    """``{side: {name, tools}}`` for ``/healthz``; tools in ``TOOLS`` order."""
    return {
        side: {
            "name": instrument.name,
            "tools": [tool for tool in TOOLS if tool in instrument.tools],
        }
        for side, instrument in instruments.items()
    }


def mask_controls(frame: dict[str, Any], instruments: Mapping[str, Instrument]) -> None:
    """Zero, in place, the trigger/grip of a validated frame for tools a side lacks."""
    for side, instrument in instruments.items():
        controller = frame.get(side)
        if not isinstance(controller, dict):
            continue
        for tool, control in _CONTROL.items():
            if tool not in instrument.tools:
                controller[control] = 0.0
