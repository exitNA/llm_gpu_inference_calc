from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gradio_ui.analysis import build_default_result  # noqa: E402
from gradio_ui.layout import launch_app  # noqa: E402
from gpu_sizing_core import format_result_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM GPU inference sizing tool")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(format_result_text(build_default_result()))
        return
    launch_app(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
