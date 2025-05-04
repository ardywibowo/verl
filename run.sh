set -euo pipefail

source "private/keys.sh"

python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

CFG=$1           # first arg = YAML file
shift            # the rest (if any) are extra Hydra overrides

# Convert YAML → dot-notation overrides
OVERRIDES=$(python3 - <<'PY' "$CFG"
import os, sys, json, yaml, re
path = sys.argv[1]
with open(path, 'r') as f:
    data = yaml.safe_load(f)

def flatten(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    else:
        # lists → JSON literal, null/bool → YAML-compatible strings
        if isinstance(node, list):
            v = json.dumps(node)
        elif node is None:
            v = "null"
        elif isinstance(node, bool):
            v = str(node).lower()
        else:
            v = str(node)
            # expand $HOME and ${VAR} patterns
            v = os.path.expandvars(v)
        yield f"{prefix}={v}"

print(" ".join(flatten(data)))
PY
)

# Launch VerL exactly like your original CLI
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    $OVERRIDES "$@"
