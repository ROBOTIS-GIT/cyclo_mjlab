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

"""Restore reset sampling on SuperMimic's concatenated motion timeline."""

from collections import Counter

import numpy as np
import torch


def sampler_snapshot(runner):
    command = getattr(runner, "motion_cmd", None)
    if command is None:
        return None
    mode = command.cfg.sampling_mode
    if mode == "adaptive":
        tensors = {
            "failures": command.bin_failed_count,
            "pending": command._current_bin_failed,
        }
    elif mode == "external":
        sampler = runner._sampler
        tensors = {
            "count": sampler.count,
            "value": sampler.value,
            "effort": sampler.effort,
            "probs": command._external_probs,
        }
    else:
        return None
    return dict(
        mode=mode,
        clips=command.motion.clip_layout,
        bins=command.bin_count,
        tensors={key: t.detach().cpu() for key, t in tensors.items()},
    )


def clip_ranges(clips):
    ranges = {}
    seen = Counter()
    start = 0
    for digest, length in clips:
        if length < 2:
            raise ValueError("Invalid checkpoint motion length")
        ranges[digest, seen[digest]] = range(start, start + length)
        seen[digest] += 1
        start += length
    return ranges, start


def remap_bins(saved, current):
    """Distribute each bin's mass over its frames, then regroup shared frames."""
    tensors = saved["tensors"]
    old_bins, bins = saved["bins"], current["bins"]
    if set(tensors) != set(current["tensors"]) or any(
        t.shape != (old_bins,) for t in tensors.values()
    ):
        raise ValueError("Invalid reset sampler checkpoint tensors")
    if saved["clips"] == current["clips"] and old_bins == bins:
        return tensors

    old_clips, old_total = clip_ranges(saved["clips"])
    clips, total = clip_ranges(current["clips"])
    source, target = [], []
    for key, frames in clips.items():
        previous = old_clips.get(key)
        if previous is None:
            continue
        if len(previous) != len(frames):
            raise ValueError("Inconsistent frame count for a shared motion")
        source.append(
            np.arange(previous.start, previous.stop) * old_bins // old_total
        )
        target.append(np.arange(frames.start, frames.stop) * bins // total)
    if not source:
        return current["tensors"]

    source, target = np.concatenate(source), np.concatenate(target)

    # ReturnSampler assigns frame f to floor(f * bin_count / frame_count).
    def widths(bin_count, frame_count):
        edges = (
            np.arange(bin_count + 1) * frame_count + bin_count - 1
        ) // bin_count
        return np.diff(edges)

    fraction = 1.0 / widths(old_bins, old_total)[source]

    def project(data):
        return np.bincount(
            target, weights=data[source] * fraction, minlength=bins
        )

    result = {}
    for key, t in tensors.items():
        if key == "value":
            continue
        values = project(t.numpy())
        if key == "probs":
            # Unseen frames start with their share of the native uniform prior.
            unseen = 1.0 - np.bincount(target, minlength=bins) / widths(
                bins, total
            )
            values += unseen / bins
            mass = values.sum()
            values = values / mass if mass > 0 else np.full(bins, 1.0 / bins)
        result[key] = torch.as_tensor(values, dtype=t.dtype)
    if "value" in tensors:
        count = result["count"]
        weighted = project((tensors["value"] * tensors["count"]).numpy())
        result["value"] = torch.where(
            count > 0,
            torch.as_tensor(weighted).to(count) / count,
            current["tensors"]["value"],
        )
    return result


def restore_sampler(runner, saved):
    if saved is None:
        if getattr(runner, "_sampler", None) is not None:
            print(
                "[INFO] Checkpoint has no reset sampler history; starting fresh."
            )
        return
    current = sampler_snapshot(runner)
    if current is None or current["mode"] != saved["mode"]:
        raise ValueError("Reset sampler mode differs from the checkpoint")
    restored = remap_bins(saved, current)
    command = runner.motion_cmd
    if saved["mode"] == "adaptive":
        command.bin_failed_count.copy_(restored["failures"])
        command._current_bin_failed.copy_(restored["pending"])
    else:
        for key in ("value", "count", "effort"):
            getattr(runner._sampler, key).copy_(restored[key])
        command.set_sampling_distribution(restored["probs"])
