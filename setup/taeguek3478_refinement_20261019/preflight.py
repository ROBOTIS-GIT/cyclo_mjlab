"""GPU integration: real reference masks, reset banks, reward caching and finite steps."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
import source.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg,CHAPTERS,FOCUS
configure_torch_backends();results=[]
for chapter in CHAPTERS:
    cfg=env_cfg(chapter);cfg.scene.num_envs=32;cfg.seed=42
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env.reset()
    cmd=env.command_manager.get_term('reference_trajectory')
    assert (cmd.frame_ids[:9]==0).all()
    focus=cmd.frame_ids[9:15].float()/float(cmd.reference.fps)
    assert all(any(a-.02<=t<b+.02 for a,b in FOCUS[chapter]) for t in focus.tolist())
    if chapter!=3:
        assert cmd.entry_bank is not None
        # Bank initialization must actually reach the physics state with named joint mapping.
        actual=cmd.robot.data.joint_pos[:4]
        bank=cmd.entry_bank['joint_pos'];limits=cmd.robot.data.soft_joint_pos_limits[0]
        bank=bank.clamp(limits[:,0],limits[:,1])
        distance=(actual[:,None,:]-bank[None]).abs().amax(-1).amin(-1)
        assert distance.max()<1e-5,distance
    name='targeted_turn' if chapter==3 else 'taeguek_support_quality'
    tc=env.reward_manager.get_term_cfg(name);term=tc.func
    coverage=[]
    if chapter!=3:
        windows={4:[(14.63,14.70,0),(4.735,4.92,1)],7:[(5.265,5.31,1),(6.71,6.755,0)],8:[(26.565,26.64,0),(38.37,38.44,0)]}[chapter]
        fps=float(cmd.reference.fps)
        for a,b,f in windows:
            fraction=term.reference_support[round(a*fps):round(b*fps),f].float().mean().item()
            assert fraction==1.,(chapter,a,b,fraction)
            coverage.append(dict(start=a,end=b,foot=f,eligible=fraction))
        assert not (term.air_mask&term.reference_support).any()
        assert term.air_mask.any()
    else:
        assert (term.turn_id==1).any() and (term.turn_id==2).any()
    for step in range(64):
        obs,reward,terminated,truncated,info=env.step(torch.zeros(32,env.action_manager.total_action_dim,device=env.device))
        assert torch.isfinite(reward).all()
        cost=term(env,**tc.params).clone();before=term.age.clone()
        torch.testing.assert_close(cost,term(env,**tc.params),rtol=0,atol=0)
        torch.testing.assert_close(before,term.age,rtol=0,atol=0)
        assert torch.isfinite(cost).all() and (cost>=0).all()
    others=term.age[2:].clone();term.reset(torch.tensor([0,1],device=env.device))
    assert (term.age[:2]==0).all()
    torch.testing.assert_close(others,term.age[2:],rtol=0,atol=0)
    row=dict(chapter=chapter,steps=64,coverage=coverage,cache_partial_reset=True,
        initial_bank_verified=chapter!=3,focus_sample_seconds=focus.tolist())
    results.append(row);print('PREFLIGHT_RESULT '+json.dumps(row),flush=True)
    env.close();del env,term,cmd;torch.cuda.empty_cache()
(HERE/'preflight.json').write_text(json.dumps(results,indent=2))
