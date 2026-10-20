"""Register only this isolated task, then use the standard training CLI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import source.tasks
from mjlab.tasks.registry import register_mjlab_task
from refinement import env_cfg, agent_cfg, SmoothRunner

TASK = 'Cyclo-Mimic-K1-Rev1-Taeguek05-SmoothV3'
register_mjlab_task(task_id=TASK, env_cfg=env_cfg(), play_env_cfg=env_cfg(True),
    rl_cfg=agent_cfg(), runner_cls=SmoothRunner)

if __name__ == '__main__':
    sys.argv.insert(1, TASK)
    from scripts.reinforcement_learning.train import main
    main()
