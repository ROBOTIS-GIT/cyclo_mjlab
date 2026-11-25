^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_mjlab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1.1 (2026-11-25)
------------------
* Fixed checkpoint resume, world-model transfer, and playback in BlinkMimic and SuperMimic.
* Corrected policy export metadata and RSL-RL action clipping in ONNX exports.
* Improved checkpoint discovery and training records, and separated BeyondMimic logs by task ID.
* Contributors: Kiwoong Park

1.1.0 (2026-11-23)
------------------
* Added SuperMimic support for world-model training and motion tracking with GenCoord and MaxCoord policies.
* Added BlinkMimic support for training and playing back motion-tracking policies from 50 Hz K1 reference clips.
* Added export of ONNX policies and ``sim2real.yaml`` deployment configurations for SuperMimic and BlinkMimic.
* Added K1 Rev.2 support across Velocity, BeyondMimic, SuperMimic, and BlinkMimic,
  and refreshed the Dance1 and Dance2 reference motions for both K1 revisions.
* Contributors: Kiwoong Park, Woojin Wie. Insu Park

1.0.0 (2026-11-16)
------------------
* Updated the runtime to MJLab 1.6.1 and Python 3.12.
* Breaking change: existing Python 3.11/MJLab 1.2 Docker images must be rebuilt before running this release.
* Updated MuJoCo and MuJoCo Warp to 3.11.0, Warp to 1.14.0, and pinned Torch to 2.14.0.
* Added hash-locked runtime dependencies in ``requirements.lock`` for Docker installation.
* Updated Docker to Python 3.12 while retaining the ``cyclo-mjlab:latest`` image and ``cyclo_mjlab`` container defaults.
* Added an image version label, runtime compatibility checks, and a container recreation command with upgrade instructions.
* Made K1 collision masks and contact priorities explicit for MJLab 1.6.1.
* Fixed Mimic partial-reset target updates, accumulated failure sampling, and state refresh after motion resampling.
* Preserved the existing flat-ground foot-height and clearance formulas and switched to the upstream ``upright`` reward.
* Updated runtime documentation and third-party attribution notices.
* Contributors: Insu Park, Woojin Wie

0.0.3 (2026-09-18)
------------------
* Updated the ``ai_sapiens`` submodule from 0.1.1 to 0.2.2.
* Updated K1 masses, centers of mass, inertias, and meshes from the upstream model.
* Fixed duplicate IMU site errors by reusing the upstream ``imu`` site.
* Contributors: Insu Park

0.0.2 (2026-09-02)
------------------
* Updated the K1 MJCF source to use the ``ai_sapiens`` submodule.
* Simplified the Mimic ONNX interface to ``obs`` input and ``actions`` output.
* Added license and attribution notices to the modified files.
* Contributors: Insu Park

0.0.1 (2026-08-28)
------------------
* Developed as an external package for MJLab and MuJoCo.
* Verified compatibility with MJLab 1.2.0.
* Verified compatibility with MuJoCo and MuJoCo Warp 3.5.0.
* Verified compatibility with Python 3.11.
* Added reinforcement learning environments for the ROBOTIS K1 Rev.1 humanoid robot.
* Added velocity-tracking locomotion support.
* Added Mimic Dance1 and Dance2 reference-motion tracking support.
* Added motion conversion and kinematics-only replay tools.
* Added a Docker development environment for training and playback.
* Contributors: Insu Park
