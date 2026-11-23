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

"""Report Docker output permission failures at the training CLI boundary."""
from contextlib import contextmanager
import os
from pathlib import Path


def _managed_path(path):
    project = Path(__file__).resolve().parents[2]
    home = Path.home()
    roots = [project / name for name in ("logs", "wandb", ".blinkmimic-data", ".supermimic-data")]
    roots.extend(home / name for name in (".cache/warp", ".cache/pip", ".cache/wandb", ".triton", ".nv/ComputeCache", ".local/share/wandb"))
    return any(path.is_relative_to(root) for root in roots)


@contextmanager
def training_permission_hint():
    try:
        yield
    except PermissionError as error:
        # Libraries such as W&B can wrap the original error and omit its filename.
        cause = error
        while cause is not None and not getattr(cause, "filename", None):
            cause = cause.__cause__ or cause.__context__
        if cause is None or not Path("/.dockerenv").exists():
            raise
        path = Path(os.fsdecode(cause.filename)).resolve(strict=False)
        if not _managed_path(path):
            raise
        reason = "Permission denied"
        location = path
        # A missing output needs write and search access on its existing parent.
        try:
            path.stat()
        except FileNotFoundError:
            for parent in path.parents:
                if parent.is_dir():
                    if os.access(parent, os.X_OK) and not os.access(parent, os.W_OK):
                        reason = "No write permission"
                        location = parent
                    break
        except PermissionError:
            pass

        raise SystemExit(
            f"\n[ERROR] Training stopped: {reason}\n"
            f"  {location}\n\n"
            "  1. Open a host terminal in the cyclo_mjlab repository and run:\n\n"
            "       ./docker/repair_permissions.sh\n\n"
            "  2. Retry the training command in this container."
        ) from None
