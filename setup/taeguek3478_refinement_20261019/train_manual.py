"""Manual split runs: resume after the last completed update, preserving optimizer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import source.tasks
from mjlab.tasks.registry import register_mjlab_task
from refinement import CHAPTERS,env_cfg,agent_cfg,TargetedRunner

class ManualRunner(TargetedRunner):
    def load(self,path,load_cfg=None,strict=True,map_location=None):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        resumed=(saved.get('infos') or {}).get('targeted_3478') is not None
        result=super().load(path,load_cfg,strict,map_location)
        if resumed:
            completed=int(saved['iter'])+1
            assert self.env.unwrapped.common_step_counter==completed*self.cfg['num_steps_per_env']
            # RSL-RL checkpoints store the last completed iteration index.
            self.current_learning_iteration=completed
            print('MANUAL_RESUME completed_updates=',completed,'optimizer_restored=True',flush=True)
        return result

for c in CHAPTERS:
    register_mjlab_task(task_id=f'Cyclo-Mimic-K1-Rev1-Taeguek{c:02d}-TargetedV3',
        env_cfg=env_cfg(c),play_env_cfg=env_cfg(c,True),rl_cfg=agent_cfg(),runner_cls=ManualRunner)
if __name__=='__main__':
    chapter=int(sys.argv[1]);sys.argv[1]=f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-TargetedV3'
    from scripts.reinforcement_learning.train import main
    main()
