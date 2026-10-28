#!/usr/bin/env python3
"""Check deployed K1 physics against a snapshot of the resolved training entity."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco as mj
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.model_dir
    ref = json.loads((root/'mjlab_physics_reference.json').read_text())
    checked, errors = 0, []
    model = mj.MjModel.from_xml_path(str(root/'k1_mjlab.xml'))
    def compare(label, actual, expected):
        nonlocal checked
        checked += 1
        if not np.allclose(actual, expected, rtol=1e-7, atol=1e-8):
            errors.append(label)
    for kind, obj in [('body',mj.mjtObj.mjOBJ_BODY),('joint',mj.mjtObj.mjOBJ_JOINT),('geom',mj.mjtObj.mjOBJ_GEOM)]:
        for name, fields in ref[kind].items():
            idx = mj.mj_name2id(model, obj, name)
            if idx < 0:
                errors.append(f'Missing {kind}: {name}')
                continue
            for field, expected in fields.items():
                compare(f'{name}.{field}', getattr(model,field)[idx], expected)
    for field in ('dof_armature','dof_damping','dof_frictionloss'):
        compare(field, getattr(model,field), ref[field])
    floor = model.geom('floor').id
    for field, expected in ref['floor'].items():
        compare('floor.'+field, getattr(model,'geom_'+field)[floor], expected)
    for name, expected in ref['mesh_sha256'].items():
        checked += 1
        if hashlib.sha256((root/'mjlab_assets_current'/name).read_bytes()).hexdigest() != expected:
            errors.append('Mesh content: '+name)
    compare('dimensions', [model.nq,model.nv,model.nu], [30,29,23])
    compare('timing/solver', [model.opt.timestep,model.opt.iterations,model.opt.ccd_iterations], [.005,10,50])
    # Every joint remains directly torque-controlled by the native PD loop.
    compare('actuator gain', model.actuator_gainprm[:,0], np.ones(model.nu))
    compare('actuator bias', model.actuator_biasprm, np.zeros_like(model.actuator_biasprm))
    for a in range(model.nu):
        j = model.actuator_trnid[a,0]
        compare(f'actuator {a} torque limit',model.actuator_ctrlrange[a], model.jnt_actfrcrange[j])
    for suffix in ('A','B','C'):
        variant=mj.MjModel.from_xml_path(str(root/f'k1_mjlab_soft_{suffix}.xml'))
        compare('soft '+suffix+' mass',variant.body_mass,model.body_mass)
        compare('soft '+suffix+' foot pairs',variant.npair,18)
    result={'passed':not errors,'checks':checked,'mismatches':errors,'mass_kg':float(model.body_mass.sum()),
            'training_mujoco':ref['mujoco_version'],'verification_mujoco':mj.__version__}
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
