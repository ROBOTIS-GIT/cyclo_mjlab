# Copyright 2025 The mjlab Developers
# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

import torch


def motion_end(env, command_name):
    command = env.command_manager.get_term(command_name)
    return torch.isin(command.time_steps + 1, command.motion.motion_ends)
