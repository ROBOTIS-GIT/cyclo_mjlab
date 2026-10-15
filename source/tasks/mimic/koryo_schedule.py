# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Versioned Koryo schedule and full-motion evaluation gate (no simulator state)."""

import math

VERSION = 2
NOISE_SCALES = (0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
PUSH_SCALES = (0.0, 0.25, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_EVAL_EPISODES = 96
MIN_EVAL_SEEDS = 3
SUCCESS_THRESHOLD = 0.9
REQUIRED_PASSES = 2


def initial_state(stage=0):
  return {
    "version": VERSION, "difficulty_stage": stage,
    "noise_scale": NOISE_SCALES[stage], "push_scale": PUSH_SCALES[stage],
    "attempts": 0, "successes": 0, "last_success_rate": 0.0,
    "evaluation_streak": 0, "last_evaluation_iteration": -1,
    "last_full_success_rate": 0.0, "mastered": False,
  }


def migrate_state(state):
  """Old final stage deliberately resumes at noise=100%, push=50%."""
  if state is None:
    return initial_state()
  version = state.get("version", 1)
  stage = state["difficulty_stage"]
  if not isinstance(stage, int) or not 0 <= stage < (4 if version == 1 else len(NOISE_SCALES)):
    raise ValueError("Invalid Koryo curriculum stage.")
  if version == 1:
    # Never interpret a legacy index as a new push level or keep its old gate.
    return initial_state(stage)
  if version != VERSION:
    raise ValueError(f"Unsupported Koryo curriculum version: {version}")
  result = {key: state[key] for key in initial_state(stage)}
  if (result["noise_scale"] != NOISE_SCALES[stage]
      or result["push_scale"] != PUSH_SCALES[stage]):
    raise ValueError("Checkpoint scales do not match the versioned Koryo schedule.")
  for key in ("attempts", "successes", "evaluation_streak", "last_evaluation_iteration"):
    if not isinstance(result[key], int):
      raise ValueError(f"Invalid curriculum counter: {key}")
  if (not 0 <= result["successes"] <= result["attempts"]
      or not 0 <= result["evaluation_streak"] <= REQUIRED_PASSES
      or result["last_evaluation_iteration"] < -1
      or not isinstance(result["mastered"], bool)
      or (result["mastered"] and (stage != len(NOISE_SCALES)-1
                                  or result["evaluation_streak"] < REQUIRED_PASSES))):
    raise ValueError("Invalid Koryo curriculum counters or completion state.")
  for key in ("last_success_rate", "last_full_success_rate"):
    if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
      raise ValueError(f"Invalid curriculum rate: {key}")
  return result


def record_full_evaluation(state, summaries, iteration):
  """Credit complete, independent-seed, mean-action full-reference evaluations."""
  result = migrate_state(state)
  if iteration <= result["last_evaluation_iteration"]:
    raise ValueError("An evaluation needs a newly trained checkpoint iteration.")
  if result["mastered"]:
    return result
  if len({s["seed"] for s in summaries}) != len(summaries) or len(summaries) < MIN_EVAL_SEEDS:
    raise ValueError("The full-motion gate requires at least three distinct seeds.")
  for s in summaries:
    if (s["condition"] != "train" or not s["full_motion_from_zero"] or not s["deterministic"]
        or s["noise_scale"] != result["noise_scale"]
        or s["push_scale"] != result["push_scale"]
        or not 0 <= s["reference_end_successes"] <= s["episodes"]
        or s["episodes"] <= 0):
      raise ValueError("Evaluation does not match the full-motion curriculum gate.")
  attempts = sum(s["episodes"] for s in summaries)
  if attempts < MIN_EVAL_EPISODES:
    raise ValueError("The full-motion gate requires at least 96 attempts.")
  rate = sum(s["reference_end_successes"] for s in summaries) / attempts
  result["last_evaluation_iteration"] = iteration
  result["last_full_success_rate"] = rate
  result["evaluation_streak"] = result["evaluation_streak"] + 1 if rate >= SUCCESS_THRESHOLD else 0
  if result["evaluation_streak"] >= REQUIRED_PASSES:
    stage = result["difficulty_stage"]
    if stage == len(NOISE_SCALES)-1:
      result["mastered"] = True
    else:
      result.update(initial_state(stage+1))
      result["last_evaluation_iteration"] = iteration
  return result
