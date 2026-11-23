"""Transfer reset-sampler history by clip identity and frame range."""

import json
from pathlib import Path


def clip_bins(directory):
    from blinkmimic.motion import BIN_FRAMES

    clips = json.loads((Path(directory) / "manifest.json").read_text())[
        "clips"
    ]
    result = {}
    offset = 0
    for clip in clips:
        name = clip["song"]
        frames = clip["frames"]
        if name in result or frames < 2 or clip["fps"] != 50:
            raise ValueError(f"Invalid BlinkMimic sampler clip: {name}")
        count = (frames + BIN_FRAMES - 1) // BIN_FRAMES
        result[name] = (
            clip["derived_sha256"],
            frames,
            range(offset, offset + count),
        )
        offset += count
    if not result:
        raise ValueError("Empty BlinkMimic motion library")
    return result


def transfer_sampler(control, source, target, *, allow_removed):
    old, new = clip_bins(source), clip_bins(target)
    if not allow_removed and (removed := old.keys() - new.keys()):
        raise ValueError(
            f"Resume task lost motion clips: {', '.join(sorted(removed))}"
        )

    source_bins, target_bins = [], []
    for name, (digest, frames, bins) in new.items():
        if name in old:
            if old[name][:2] != (digest, frames):
                raise ValueError(
                    f"Resume motion changed under the same name: {name}"
                )
            source_bins.extend(old[name][2])
            target_bins.extend(bins)
    if control is None or old == new:
        return control

    old_size = sum(len(row[2]) for row in old.values())
    size = sum(len(row[2]) for row in new.values())

    def transfer(values, fill=0.0):
        if values.shape != (old_size,):
            raise ValueError(
                "Checkpoint sampler size differs from the source manifest"
            )
        result = values.new_full((size,), fill)
        result[target_bins] = values[source_bins]
        return result

    bins = control["env_bins"]
    weights = transfer(bins["bin_w_avg"], 1.0 / old_size)
    restored = dict(
        control,
        env_bins={
            "bin_G": transfer(bins["bin_G"]),
            "bin_fail": transfer(bins["bin_fail"]),
            "bin_w_avg": weights / weights.sum(),
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
    return restored
