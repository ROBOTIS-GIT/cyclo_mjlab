"""SuperMimic compatibility imports for the shared K1 reference adapter."""

from source.utils.k1_motion_npz_adapter import (  # noqa: F401
    ADAPTER_VERSION,
    BODY_FIELDS,
    FIELDS,
    ROOT_CONTRACT,
    adapt_motion,
    convert_motion,
    model_joint_names,
    prepare_motion_files,
    validate_motion,
)
