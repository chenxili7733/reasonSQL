#!/usr/bin/env python3
"""Apply the optional ReasonSQL OPD coefficient-schedule patch to veRL v0.8.0.

veRL v0.8.0 already implements teacher serving, token alignment and OPD losses.
This patch only adds a per-step ``distillation_loss_coef`` supplied in the
training TensorDict.  It intentionally does not replace any native OPD logic.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


MARKER = "ReasonSQL OPD schedule patch v1"


@dataclass(frozen=True)
class Patch:
    relative_path: str
    description: str
    anchor: re.Pattern[str]
    insertion: str
    before: bool = False


PATCHES = (
    Patch(
        relative_path="verl/trainer/ppo/ray_trainer.py",
        description="attach scheduled OPD coefficient to each actor batch",
        anchor=re.compile(r"^(\s*)batch_td = batch\.to_tensordict\(\)\s*$"),
        insertion=(
            "{indent}# {marker}\n"
            "{indent}if self.config.get(\"distillation\", {{}}).get(\"enabled\", False):\n"
            "{indent}    from reasonsql.opd.schedule import coefficient_from_env\n"
            "{indent}\n"
            "{indent}    tu.assign_non_tensor(\n"
            "{indent}        batch_td,\n"
            "{indent}        distillation_loss_coef=coefficient_from_env(\n"
            "{indent}            self.global_steps, self.total_training_steps\n"
            "{indent}        ),\n"
            "{indent}    )\n"
        ),
    ),
    Patch(
        relative_path="verl/trainer/distillation/losses.py",
        description="consume the scheduled OPD coefficient in the native loss",
        anchor=re.compile(r"^(\s*)policy_loss \+= distill_loss \* distillation_loss_coef\s*$"),
        insertion=(
            "{indent}# {marker}\n"
            "{indent}scheduled_coef = data.get(\"distillation_loss_coef\", None)\n"
            "{indent}if scheduled_coef is not None:\n"
            "{indent}    if hasattr(scheduled_coef, \"item\"):\n"
            "{indent}        scheduled_coef = scheduled_coef.item()\n"
            "{indent}    distillation_loss_coef = float(scheduled_coef)\n"
            "{indent}    policy_metrics[\"distillation/loss_coef\"] = Metric(\n"
            "{indent}        value=distillation_loss_coef, aggregation=AggregationType.MEAN\n"
            "{indent}    )\n"
        ),
        before=True,
    ),
)


def _version_text(verl_root: Path) -> str:
    candidates = (verl_root / "setup.py", verl_root / "pyproject.toml", verl_root / "verl" / "__init__.py")
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in candidates if path.exists())


def verify_version(verl_root: Path, allow_unknown: bool = False) -> None:
    text = _version_text(verl_root)
    if "0.8.0" not in text and "v0.8.0" not in text and not allow_unknown:
        raise RuntimeError(
            "the schedule patch is pinned to veRL v0.8.0; checkout tag v0.8.0 "
            "or pass --allow-unknown-version after manually reviewing both anchors"
        )


def apply_one(root: Path, patch: Patch, dry_run: bool) -> bool:
    path = root / patch.relative_path
    if not path.is_file():
        print(f"[ERROR] missing {path}")
        return False
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"[SKIP] {patch.description}: already applied")
        return True
    lines = source.splitlines()
    matches = [(index, patch.anchor.match(line)) for index, line in enumerate(lines)]
    matches = [(index, match) for index, match in matches if match]
    if len(matches) != 1:
        print(f"[ERROR] {patch.description}: expected one anchor, found {len(matches)}")
        return False
    index, match = matches[0]
    indent = match.group(1)
    insertion = patch.insertion.format(indent=indent, marker=MARKER).rstrip().splitlines()
    if patch.before:
        new_lines = lines[:index] + insertion + lines[index:]
    else:
        new_lines = lines[: index + 1] + insertion + lines[index + 1 :]
    candidate = "\n".join(new_lines) + "\n"
    ast.parse(candidate)
    if not dry_run:
        backup = path.with_suffix(path.suffix + ".reasonsql.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(candidate, encoding="utf-8")
    print(f"[OK] {patch.description}{' (dry-run)' if dry_run else ''}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-src", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unknown-version", action="store_true")
    args = parser.parse_args()
    root = args.verl_src.expanduser().resolve()
    if not (root / "verl").is_dir():
        raise SystemExit(f"{root} does not contain a verl/ package")
    verify_version(root, args.allow_unknown_version)
    ok = all(apply_one(root, patch, args.dry_run) for patch in PATCHES)
    if not ok:
        raise SystemExit(1)
    print("ReasonSQL OPD schedule patch is ready.")


if __name__ == "__main__":
    main()
