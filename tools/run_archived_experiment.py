"""Run a frozen baseline entry point with archived project paths mapped locally."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission_duration.portable_paths import ArchivedPath

FROZEN = ROOT / ".fieldvision_audit_v14_2/0a8ac5415374654c"
RUNNERS = {"run_learning_baselines_v9", "run_orchestration_experiments"}


def load_runner(name):
    if name not in RUNNERS:
        raise ValueError(f"Unknown frozen runner: {name}")
    sys.path.insert(0, str(FROZEN))
    environment = importlib.import_module("fieldvision_noms_environment")
    runner = importlib.import_module(name)
    for module in (environment, runner):
        module.Path = ArchivedPath
    return runner


if __name__ == "__main__":
    name = sys.argv.pop(1)
    load_runner(name).main()
