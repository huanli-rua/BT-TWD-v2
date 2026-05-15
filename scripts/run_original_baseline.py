"""运行第一篇原方法最小 baseline。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bttwdlib.baseline.original_bttwd import run_original_bttwd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第一篇 BT-TWD/BSM 原方法 baseline")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "adult_bttwd.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    out_dir = REPO_ROOT / "outputs" / "original_baseline" / cfg_path.stem
    run_original_bttwd(cfg_path, cfg_override={"OUTPUT": {"results_dir": str(out_dir)}})


if __name__ == "__main__":
    main()
