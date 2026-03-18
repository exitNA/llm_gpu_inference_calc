from __future__ import annotations

import warnings

from ui.common import APP_CSS

from .analysis import build_default_result
from .bindings import bind_events
from .constants import APP_TITLE
from .config import default_component_values
from .runtime import build_theme, import_gradio
from .sections import build_results_panel, build_sidebar

demo = None


def build_app():
    gr = import_gradio()
    defaults = default_component_values()
    default_result = build_default_result()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The parameters have been moved from the Blocks constructor to the launch\(\) method in Gradio 6\.0: theme, css\..*",
            category=UserWarning,
        )
        blocks = gr.Blocks(title=APP_TITLE, theme=build_theme(gr), css=APP_CSS)

    with blocks as demo:
        with gr.Row(elem_classes=["workspace-main"]):
            sidebar = build_sidebar(gr, defaults)
            results = build_results_panel(gr, default_result)

        bind_events(sidebar=sidebar, results=results, demo=demo)

    return demo


def get_demo():
    global demo
    if demo is not None:
        return demo
    try:
        demo = build_app()
    except RuntimeError:
        demo = None
    return demo


def launch_app(host: str, port: int, share: bool) -> None:
    app = get_demo()
    if app is None:
        raise RuntimeError("未安装 gradio，请先执行 `uv sync`。")
    app.launch(server_name=host, server_port=port, share=share)
