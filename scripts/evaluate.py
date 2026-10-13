"""Run:  python -m scripts.evaluate"""
import json
from pathlib import Path

from app.evaluation import run_eval

if __name__ == "__main__":
    report = run_eval(Path("data/eval/eval_set.json"))
    print(json.dumps(report, indent=2))
