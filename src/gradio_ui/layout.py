from __future__ import annotations

from collections.abc import Callable
import warnings

from ui.common import APP_CSS

from .bindings import bind_events
from .constants import APP_TITLE
from .config import default_component_values
from .runtime import build_theme, import_gradio
from .sections import build_results_panel, build_sidebar

demo = None


def build_app(progress: Callable[[str], None] | None = None):
    if progress is not None:
        progress("导入 Gradio 运行时")
    gr = import_gradio()
    if progress is not None:
        progress("加载默认输入配置")
    defaults = default_component_values()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The parameters have been moved from the Blocks constructor to the launch\(\) method in Gradio 6\.0: theme, css\..*",
            category=UserWarning,
        )
        if progress is not None:
            progress("创建页面容器")
        blocks = gr.Blocks(title=APP_TITLE, theme=build_theme(gr), css=APP_CSS)

    if progress is not None:
        progress("组装页面组件")
    with blocks as demo:
        with gr.Row(elem_classes=["workspace-main"]):
            sidebar = build_sidebar(gr, defaults)
            results = build_results_panel(gr)

        if progress is not None:
            progress("绑定交互事件")
        bind_events(sidebar=sidebar, results=results, demo=demo)

    return demo


def get_demo(progress: Callable[[str], None] | None = None):
    global demo
    if demo is not None:
        if progress is not None:
            progress("复用已初始化页面")
        return demo
    try:
        demo = build_app(progress=progress)
    except RuntimeError:
        demo = None
    return demo


def launch_app(
    host: str,
    port: int,
    share: bool,
    progress: Callable[[str], None] | None = None,
) -> None:
    app = get_demo(progress=progress)
    if app is None:
        raise RuntimeError("未安装 gradio，请先执行 `uv sync`。")
    if progress is not None:
        progress(f"启动 Web 服务 http://{host}:{port}")
    app.launch(server_name=host, server_port=port, share=share)
