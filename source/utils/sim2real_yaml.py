# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Insu Park, Kiwoong Park

"""YAML presentation shared by policy configuration exporters."""

from typing import Any

import yaml


class _FlowList(list):
  """YAML list rendered in flow style, matching the ROBOTIS config format."""


class _Sim2RealYamlDumper(yaml.SafeDumper):
  pass


def _represent_flow_list(
  dumper: yaml.SafeDumper, data: _FlowList
) -> yaml.SequenceNode:
  return dumper.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True
  )


_Sim2RealYamlDumper.add_representer(_FlowList, _represent_flow_list)

_FLOW_LIST_KEYS = {
  "scale",
  "offset",
  "clip",
  "lin_vel_x",
  "lin_vel_y",
  "ang_vel_z",
  "heading",
}


def _format_yaml(value: Any, key: str | None = None) -> Any:
  if isinstance(value, float):
    return float(f"{value:.3g}")
  if isinstance(value, (list, tuple)):
    items = [_format_yaml(item) for item in value]
    return _FlowList(items) if key in _FLOW_LIST_KEYS else items
  if isinstance(value, dict):
    return {
      item_key: _format_yaml(item, item_key)
      for item_key, item in value.items()
    }
  return value


def dump_sim2real_yaml(config: dict) -> str:
  """Keep mappings readable and numeric setting arrays in bracket notation."""
  return yaml.dump(
    _format_yaml(config),
    Dumper=_Sim2RealYamlDumper,
    sort_keys=False,
    default_flow_style=False,
    allow_unicode=True,
  )
