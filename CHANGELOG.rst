^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_mjlab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
0.0.2 (2026-09-02)
------------------
### K1 MJCF Integration and ONNX Export Update
- Updated the K1 MJCF source to use the ``ai_sapiens`` submodule.
- Simplified the Mimic ONNX interface to ``obs`` input and ``actions`` output.
- Added license and attribution notices to the modified files.
- Contributors: Insu Park

0.0.1 (2026-08-28)
------------------
### Initial Release
- Developed as an external package for MJLab and MuJoCo.
- Verified compatibility with the following environments:
  - MJLab 1.2.0
  - MuJoCo and MuJoCo Warp 3.5.0
  - Python 3.11
- Introduced reinforcement learning environments for the ROBOTIS K1 Rev.1
  humanoid robot:
  - Velocity-tracking locomotion
  - Mimic Dance1 and Dance2 reference-motion tracking
- Added motion conversion and kinematics-only replay tools.
- Added a Docker development environment for training and playback.
- Contributors: Insu Park
