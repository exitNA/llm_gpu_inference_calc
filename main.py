import argparse
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU sizing Gradio app")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Gradio server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the built-in example once and print the result without launching Gradio",
    )
    return parser.parse_args()


def serve_app(host: str, port: int, share: bool) -> None:
    from gradio_app import launch_app

    launch_app(host=host, port=port, share=share)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        from gradio_app import build_default_result_text

        print(build_default_result_text())
        return

    serve_app(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
