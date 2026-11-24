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
# Author: Kiwoong Park

"""Transfer reset-sampler history by clip identity and frame range."""

import hashlib
import json
from pathlib import Path


def sampler_clips(directory):
    """Keep the manifest order used by ExactMotionLib, without its filesystem paths."""
    return _clip_identities((Path(directory) / "manifest.json").read_bytes())


def _clip_identities(data):
    rows = json.loads(data)["clips"]
    if any(row["fps"] != 50 for row in rows):
        raise ValueError("BlinkMimic sampler clips must be 50 Hz")
    return [(row["song"], row["derived_sha256"], row["frames"]) for row in rows]


def legacy_sampler_clips(checkpoint):
    """Recover clip order from the original manifest, including host/container remaps."""
    from ...runtime import ROOT

    record_path = Path(checkpoint).parent / "run-record.json"
    try:
        record = json.loads(record_path.read_bytes())
        manifest = Path(record["manifest"])
    except (OSError, ValueError, TypeError, KeyError):
        return None

    paths = [manifest]
    if "cwd" in record and manifest.is_relative_to(record["cwd"]):
        paths.append(ROOT / manifest.relative_to(record["cwd"]))
    # A relocated task run can still carry its original motions/manifest.json.
    if manifest.parent.name == "motions" and record.get("manifest_sha256"):
        paths.append(record_path.parent / "motions" / "manifest.json")
    digest = record.get("manifest_sha256")
    for path in dict.fromkeys(paths):
        try:
            data = path.read_bytes()
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        if digest and hashlib.sha256(data).hexdigest() != digest:
            continue
        return _clip_identities(data)
    return None


def clip_bins(clips):
    from blinkmimic.motion import BIN_FRAMES

    result = {}
    offset = 0
    for name, digest, frames in clips:
        if name in result or type(frames) is not int or frames < 2:
            raise ValueError(f"Invalid BlinkMimic sampler clip: {name}")
        count = (frames + BIN_FRAMES - 1) // BIN_FRAMES
        result[name] = (digest, frames, range(offset, offset + count))
        offset += count
    if not result:
        raise ValueError("Empty BlinkMimic motion library")
    return result


def transfer_sampler(control, source_clips, target_clips, history):
    if control is None:
        return None
    new = clip_bins(target_clips)
    old = clip_bins(source_clips) if source_clips is not None else {}
    if not old:
        print("[WARN] Original source motion manifest unavailable. Resetting BlinkMimic sampler "
              "statistics for unidentified clips.", flush=True)
    source_bins, target_bins = [], []
    returning = []
    for name, (digest, frames, bins) in new.items():
        if name in old and old[name][:2] == (digest, frames):
            source_bins.extend(old[name][2])
            target_bins.extend(bins)
        elif name not in old:
            returning.append(((name, digest, frames), bins))
        else:
            # Replacing an active clip starts its new revision with fresh statistics.
            history.pop((name, digest, frames), None)
    if old == new:
        return control
    old_size = sum(len(row[2]) for row in old.values())
    size = sum(len(row[2]) for row in new.values())

    def transfer(values, fill=0.0):
        if old and values.shape != (old_size,):
            raise ValueError(
                "Checkpoint sampler size differs from the source manifest"
            )
        result = values.new_full((size,), fill)
        if source_bins:
            result[target_bins] = values[source_bins]
        return result

    bins = control["env_bins"]
    weights = transfer(bins["bin_w_avg"], 1.0 / (old_size or size))
    mass = weights.sum()
    weights = weights / mass if mass > 0 else weights.fill_(1.0 / size)
    restored = dict(
        control,
        clips=target_clips,
        env_bins={
            "bin_G": transfer(bins["bin_G"]),
            "bin_fail": transfer(bins["bin_fail"]),
            "bin_w_avg": weights,
        },
    )
    if control.get("ep_fail") is not None:
        restored["ep_fail"] = transfer(control["ep_fail"])
    if control.get("focus_q") is not None:
        focus = transfer(control["focus_q"])
        mass = focus.sum()
        restored["focus_q"] = focus / mass if mass > 0 else None
        if restored["focus_q"] is None:
            restored["focus_share"] = 0.0
    # Return/failure measurements belong to individual bins. Sampling and
    # focus probabilities belong to the active library and stay out of history.
    for name, (digest, frames, bins) in old.items():
        if name in new:
            continue
        section = slice(bins.start, bins.stop)
        entry = {
            key: control["env_bins"][key][section].detach().to("cpu", copy=True)
            for key in ("bin_G", "bin_fail")
        }
        if control.get("ep_fail") is not None:
            entry["ep_fail"] = control["ep_fail"][section].detach().to("cpu", copy=True)
        history[name, digest, frames] = entry
    for identity, bins in returning:
        entry = history.pop(identity, None)
        if entry is None:
            continue
        section = slice(bins.start, bins.stop)
        for key in ("bin_G", "bin_fail"):
            restored["env_bins"][key][section] = entry[key].to(restored["env_bins"][key])
        if "ep_fail" in entry and restored.get("ep_fail") is not None:
            restored["ep_fail"][section] = entry["ep_fail"].to(restored["ep_fail"])
    return restored
