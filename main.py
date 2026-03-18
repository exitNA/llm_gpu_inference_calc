from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gradio_ui.analysis import build_default_result  # noqa: E402
from gradio_ui.layout import launch_app  # noqa: E402
from gpu_sizing_core import format_result_text  # noqa: E402


def startup_logger() -> callable:
    start = time.perf_counter()

    def log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[startup +{elapsed:5.1f}s] {message}...", file=sys.stderr, flush=True)

    return log


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
        print(format_result_text(build_default_result(emit_console_trace=False)))
        return
    log = startup_logger()
    log("准备启动")
    print(
        "[startup] 首次启动可能需要约 30-40s，主要耗时通常在 Gradio 冷启动",
        file=sys.stderr,
        flush=True,
    )
    launch_app(host=args.host, port=args.port, share=args.share, progress=log)


if __name__ == "__main__":
    main()
