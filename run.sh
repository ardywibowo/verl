#!/usr/bin/env bash
#
# Usage:
#   conda activate iris
#   ./run_from_yaml.sh gsm8k.yaml                   # run
#   ./run_from_yaml.sh gsm8k.yaml trainer.total_epochs=5  # plus extra Hydra overrides
#
# Requires: python3, PyYAML (`pip install pyyaml`)
set -euo pipefail

CFG=$1          # first arg = YAML file
shift           # any extra CLI overrides

OVERRIDES=$(python3 - <<'PY' "$CFG"
import os, sys, yaml, re
path = sys.argv[1]
with open(path) as f:
    data = yaml.safe_load(f)

def encode_list(lst):
    """Hydra override grammar: key=[a,b,c]  (no spaces, no quotes)."""
    items = []
    for x in lst:
        if isinstance(x, str):
            # expand $HOME etc. inside items, keep bare tokens if clean
            x = os.path.expandvars(x)
            # quote only if the token has chars Hydra treats specially
            if re.search(r'[,\[\]=]', x):
                x = f'"{x}"'
        elif x is None:
            x = "null"
        elif isinstance(x, bool):
            x = str(x).lower()
        else:
            x = str(x)
        items.append(x)
    return f"[{','.join(items)}]"

def flatten(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    else:
        if isinstance(node, list):
            v = encode_list(node)
        elif node is None:
            v = "null"
        elif isinstance(node, bool):
            v = str(node).lower()
        else:
            v = os.path.expandvars(str(node))
        yield f"{prefix}={v}"

print(" ".join(flatten(data)))
PY
)

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    $OVERRIDES "$@"
