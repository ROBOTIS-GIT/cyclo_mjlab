# Copyright 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Historical K1 COM offsets, not a replacement simulation model.

Extracted from cyclo_mjlab commit 874c7bd, file
source/assets/robots/robotis_k1/xmls/k1.xml (Apache-2.0).
Only accept this profile after all named poses and velocities match numerically.
"""

LEGACY_K1_COM = {
    "pelvis": (-0.00292649, 1.12776e-06, -0.0503154),
    "left_hip_pitch_link": (-0.00114167, 0.0794843, -0.0257981),
    "left_hip_roll_link": (0.0145172, -2.95988e-06, -0.0790822),
    "left_hip_yaw_link": (-0.00198177, -0.00493634, -0.181142),
    "left_knee_link": (-0.000387029, 0.00314811, -0.12388),
    "left_ankle_pitch_link": (0.00503247, -3.76608e-05, 0.00240232),
    "left_ankle_roll_link": (0.0346041, 4.98757e-06, -0.0409511),
    "right_hip_pitch_link": (-0.00114135, -0.0794886, -0.0257925),
    "right_hip_roll_link": (0.0145172, -2.95988e-06, -0.0790822),
    "right_hip_yaw_link": (-0.0019798, 0.00493716, -0.181143),
    "right_knee_link": (-0.000387029, -0.00314811, -0.12388),
    "right_ankle_pitch_link": (0.00503247, -3.76608e-05, 0.00240232),
    "right_ankle_roll_link": (0.0346041, 4.98757e-06, -0.0409511),
    "torso_link": (-0.0139656, 0.000601382, 0.211074),
    "head_link": (0.006953, 3.24285e-06, 0.106553),
    "left_shoulder_pitch_link": (-0.00286054, 0.0735648, 4.02891e-07),
    "left_shoulder_roll_link": (-0.000859887, -1.40133e-06, -0.0626681),
    "left_shoulder_yaw_link": (0.0124097, -0.00401596, -0.103721),
    "left_elbow_link": (0.0655204, 0.00412184, 3.49023e-06),
    "left_wrist_roll_rubber_hand": (0.140765, 0.000524863, 0.000403934),
    "right_shoulder_pitch_link": (-0.00286054, -0.0735631, 7.49207e-06),
    "right_shoulder_roll_link": (-0.000859887, -1.40133e-06, -0.0626681),
    "right_shoulder_yaw_link": (0.0124097, 0.00401596, -0.103721),
    "right_elbow_link": (0.0655204, -0.00412184, -3.49023e-06),
    "right_wrist_roll_rubber_hand": (0.140765, -0.000524858, 0.000403934),
}
