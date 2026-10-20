from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import source.tasks
from refinement import register_tasks
register_tasks()
from scripts.reinforcement_learning.train import main
if __name__=='__main__':
    chapter=int(sys.argv[1]);sys.argv[1]=f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-TargetedV3'
    main()
