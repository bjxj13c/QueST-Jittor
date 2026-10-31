import argparse
import json
from pathlib import Path
from quest_repro.batch_plan import create_plan

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--seed", type=int, default=10000)
args = parser.parse_args()
print(json.dumps(create_plan(args.root, args.output, args.epochs, args.batch_size, args.seed), indent=2))
