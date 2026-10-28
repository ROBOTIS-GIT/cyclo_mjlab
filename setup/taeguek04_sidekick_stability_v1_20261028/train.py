import os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/t6-balanced-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import source.tasks
from mjlab.tasks.registry import register_mjlab_task
from refinement import env_cfg,agent_cfg,BalancedRunner,validate_inputs
TASK='Cyclo-Mimic-K1-Rev1-Taeguek04-SidekickStabilityV1'
validate_inputs()
register_mjlab_task(task_id=TASK,env_cfg=env_cfg(),play_env_cfg=env_cfg(True),rl_cfg=agent_cfg(),runner_cls=BalancedRunner)
if __name__=='__main__':
    if '--checkpoint' not in sys.argv:raise ValueError('Use launch.py; explicit source or resume checkpoint required')
    sys.argv.insert(1,TASK)
    from scripts.reinforcement_learning.train import main
    main()
