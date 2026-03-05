"""
experiments/phase2/build_manifest.py
-------------------------------------
Build test manifest + pre-tokenised prompts.  Run ONCE on the GPU server.

Outputs (all under data/exp/phase2/data/):
  test_manifest.json        — routing metadata for every test row
  prompts_venra.jsonl       — Qwen-format prompts ending with "Label:"
  prompts_frontier.jsonl    — Qwen-tag-stripped prompts for API models

Reads from:
  data/training_final/test.jsonl   (primary)
  pagand/venra@v2.3  via HuggingFace (fallback, also saves locally)

Usage:
  python -m experiments.phase2.build_manifest
"""

import json
import os
import re
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer

# Import build_prompt — matches train.py import exactly
try:
    from src.hal_det.prompt_builder import build_prompt
except ImportError:
    # Fallback if src is already in sys.path
    from hal_det.prompt_builder import build_prompt

from experiments.phase2.utils import (
    ensure_dirs,
    MANIFEST_PATH, PROMPTS_VENRA_PATH, PROMPTS_FRONTIER_PATH,
    SABOTAGE_TYPES, ALL_PAIR_TAGS,
    # NOTE: PROJECT_ROOT is defined locally above; do NOT import it from utils
    # (both resolve to the same path but the import would shadow the local def)
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_TEST_PATH  = PROJECT_ROOT / "data" / "training_final" / "test.jsonl"
HF_DATASET       = "pagand/venra"
HF_REVISION      = "v2.3"
MAX_SEQ_LENGTH   = 4096
LENGTH_THRESHOLD = 512      # same as SabotageEvalCallback
COT_PAIRS        = 50       # number of pairs for CoT subsample
SEED             = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data() -> List[Dict]:
    if LOCAL_TEST_PATH.exists():
        print(f"[data] Loading from {LOCAL_TEST_PATH}")
        rows = []
        with open(LOCAL_TEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"[data] Loaded {len(rows)} rows from local file")
        return rows

    print(f"[data] Local file not found — fetching from HuggingFace {HF_DATASET}@{HF_REVISION}")
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, revision=HF_REVISION, split="test",
                      token=os.environ.get("HF_TOKEN"))
    rows = [dict(r) for r in ds]
    print(f"[data] Downloaded {len(rows)} rows")

    # Persist locally for future runs
    LOCAL_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_TEST_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[data] Saved to {LOCAL_TEST_PATH}")
    return rows


# ── Family routing — mirrors SabotageEvalCallback.__init__ exactly ─────────────

def route_families(
    rows: List[Dict],
) -> Tuple[List, List, List, List, List]:
    """
    Returns: pairs_short, pairs_long, natural_fake, natural_true, axiom_pool
    Each pair is (parent_row, child_row).
    sabotage_type is a TOP-LEVEL field (not inside meta) — same as train.py.
    """
    families: Dict[str, List] = defaultdict(list)
    for row in rows:
        fid = row.get("meta", {}).get("family_id")
        if fid:
            families[fid].append(row)

    pairs_short:   List[Tuple] = []
    pairs_long:    List[Tuple] = []
    natural_fake:  List[Dict]  = []
    natural_true:  List[Dict]  = []
    axiom_pool:    List[Dict]  = []
    discarded = 0

    for fid, members in sorted(families.items()):
        supported = [m for m in members if m["label"] == "Supported"]
        unfounded = [m for m in members if m["label"] == "Unfounded"]
        general   = [m for m in members if m["label"] == "General"]

        axiom_pool.extend(general)

        if supported and unfounded:
            parent = supported[0]
            child  = unfounded[0]
            t_count = parent.get("meta", {}).get("token_count", 0)
            pair = (parent, child)
            if t_count >= LENGTH_THRESHOLD:
                pairs_long.append(pair)
            else:
                pairs_short.append(pair)
        else:
            for m in supported:
                natural_true.append(m)
            for m in unfounded:
                # sabotage_type is a root-level field
                if m.get("sabotage_type", "unknown") == "natural":
                    natural_fake.append(m)
                else:
                    discarded += 1
                    fid_str = m.get("meta", {}).get("family_id", "?")
                    st = m.get("sabotage_type", "?")
                    print(f"  [warn] Orphaned non-natural Unfounded discarded: "
                          f"family={fid_str}, sabotage_type={st}")

    print(f"\n[route] Pool sizes:")
    print(f"  pairs_short  (token_count < {LENGTH_THRESHOLD}): {len(pairs_short)}")
    print(f"  pairs_long   (token_count >= {LENGTH_THRESHOLD}): {len(pairs_long)}")
    print(f"  natural_fake: {len(natural_fake)}")
    print(f"  natural_true: {len(natural_true)}")
    print(f"  axioms:        {len(axiom_pool)}")
    print(f"  discarded:     {discarded}")

    return pairs_short, pairs_long, natural_fake, natural_true, axiom_pool


# ── CoT subsample selection ───────────────────────────────────────────────────

def select_cot_subsample(
    pairs_short: List[Tuple],
    pairs_long:  List[Tuple],
    n_pairs: int = COT_PAIRS,
    seed: int = SEED,
) -> set:
    """
    Select n_pairs stratified by child's sabotage_type.
    Returns a set of family_ids for the selected pairs.
    """
    rng = random.Random(seed)
    all_pairs = pairs_short + pairs_long

    by_type: Dict[str, List] = defaultdict(list)
    for parent, child in all_pairs:
        st = child.get("sabotage_type", "unknown")
        if st in SABOTAGE_TYPES:
            by_type[st].append((parent, child))

    selected = []
    per_type  = n_pairs // len(SABOTAGE_TYPES)
    remainder = n_pairs - per_type * len(SABOTAGE_TYPES)

    for i, st in enumerate(SABOTAGE_TYPES):
        take = per_type + (1 if i < remainder else 0)
        pool = list(by_type[st])
        rng.shuffle(pool)
        selected.extend(pool[:take])

    fids = {
        parent.get("meta", {}).get("family_id")
        for parent, child in selected
    }
    print(f"\n[subsample] CoT subsample: {len(selected)} pairs "
          f"({[len(by_type[st]) for st in SABOTAGE_TYPES]} per type)")
    return fids


# ── Frontier prompt extraction ────────────────────────────────────────────────

def strip_qwen_tags_for_frontier(venra_prompt: str) -> Tuple[str, str]:
    """
    Extract (system_content, user_content) from a Qwen-formatted prompt.
    user_content retains the trailing 'Label:' to condition first-token output.
    """
    # System turn
    sys_match = re.search(
        r'<\|im_start\|>system\n(.*?)<\|im_end\|>',
        venra_prompt, re.DOTALL
    )
    system_content = sys_match.group(1).strip() if sys_match else \
        "You are a rigorous financial auditor."

    # User turn (everything before the assistant turn)
    user_match = re.search(
        r'<\|im_start\|>user\n(.*?)<\|im_end\|>\s*<\|im_start\|>assistant',
        venra_prompt, re.DOTALL
    )
    user_content = user_match.group(1).strip() if user_match else venra_prompt
    user_content = user_content + "\nLabel:"   # condition next token

    return system_content, user_content


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()

    # Tokenizer
    print(f"[tokenizer] Loading: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Data
    rows = load_test_data()

    # Route
    pairs_short, pairs_long, natural_fake, natural_true, axiom_pool = \
        route_families(rows)

    # CoT subsample
    cot_family_ids = select_cot_subsample(pairs_short, pairs_long)

    # ── Build flat layout ──────────────────────────────────────────────────────
    # Order: [pair_short | pair_long | natural_fake | natural_true | axiom]
    # Within each pair: parent immediately before child (row_id offset = +1).
    manifest:         List[Dict] = []
    venra_prompts:    List[Dict] = []
    frontier_prompts: List[Dict] = []
    row_id = 0

    def add_row(row: Dict, pool_tag: str, gt_int: int) -> None:
        nonlocal row_id

        fid = row.get("meta", {}).get("family_id", "")
        # cot_subsample must only be True for PAIR rows whose family was selected.
        # Axiom (General) rows can share a family_id with a pair — if we set
        # cot_subsample=True on them, compute_metrics(use_subsample=True) would
        # include axiom rows in the subsample manifest, producing empty predictions
        # for them (the CoT script only runs on pairs) and making acc_axiom=0,
        # which drives composite=0 for every model in the subsample comparison.
        is_cot = (fid in cot_family_ids) and (pool_tag in ALL_PAIR_TAGS)

        c_raw   = row["input_components"]["context"]
        context = "\n".join(c_raw) if isinstance(c_raw, list) else str(c_raw)

        # Build VeNRA prompt (Qwen format, ends with "Label:")
        prompt_text = build_prompt(
            query          = row["input_components"]["query"],
            context        = context,
            trace          = row["input_components"]["trace"],
            statement      = row["output_components"]["target_sentence"],
            tokenizer      = tokenizer,
            max_seq_length = MAX_SEQ_LENGTH,
            label_token    = None,   # inference mode
            reasoning      = None,
            prompt_type    = "full",
        )

        # Frontier prompt
        system_content, user_content = strip_qwen_tags_for_frontier(prompt_text)

        manifest.append({
            "row_id":       row_id,
            "family_id":    fid,
            "label":        row["label"],
            "ground_truth": gt_int,
            "sabotage_type": row.get("sabotage_type", "natural"),
            "pool":         pool_tag,
            "meta_token_count": row.get("meta", {}).get("token_count", 0),
            "cot_subsample": is_cot,
            # Keep source data so CoT scripts can rebuild prompts if needed
            "input_components":  row["input_components"],
            "output_components": row["output_components"],
        })

        tok_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        venra_prompts.append({
            "row_id":      row_id,
            "prompt_text": prompt_text,
            "token_count": tok_count,
        })

        frontier_prompts.append({
            "row_id":         row_id,
            "system_content": system_content,
            "user_content":   user_content,
        })

        row_id += 1

    # Pairs short
    for parent, child in pairs_short:
        add_row(parent, "pair_short_parent", 0)
        add_row(child,  "pair_short_child",  1)

    # Pairs long
    for parent, child in pairs_long:
        add_row(parent, "pair_long_parent", 0)
        add_row(child,  "pair_long_child",  1)

    # Natural fake
    for row in natural_fake:
        add_row(row, "natural_fake", 1)

    # Natural true
    for row in natural_true:
        add_row(row, "natural_true", 0)

    # Axioms
    for row in axiom_pool:
        add_row(row, "axiom", 2)

    # ── Persist ───────────────────────────────────────────────────────────────
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[save] Manifest → {MANIFEST_PATH}  ({len(manifest)} rows)")

    with open(PROMPTS_VENRA_PATH, "w") as f:
        for r in venra_prompts:
            f.write(json.dumps(r) + "\n")
    print(f"[save] VeNRA prompts → {PROMPTS_VENRA_PATH}")

    with open(PROMPTS_FRONTIER_PATH, "w") as f:
        for r in frontier_prompts:
            f.write(json.dumps(r) + "\n")
    print(f"[save] Frontier prompts → {PROMPTS_FRONTIER_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    pool_counts = Counter(r["pool"] for r in manifest)
    cot_count   = sum(1 for r in manifest
                      if r["cot_subsample"] and r["pool"].endswith("parent"))

    print(f"\n[summary] Total rows: {len(manifest)}")
    print(f"[summary] CoT subsample pairs: {cot_count}")
    print(f"[summary] Pool breakdown:")
    for pool_tag in [
        "pair_short_parent", "pair_short_child",
        "pair_long_parent",  "pair_long_child",
        "natural_fake", "natural_true", "axiom",
    ]:
        print(f"    {pool_tag:<25}: {pool_counts.get(pool_tag, 0)}")


if __name__ == "__main__":
    main()