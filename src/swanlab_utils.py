
def build_swanlab_callback(cfg, run_name):
    """构建SwanLab回调；返回None表示本次不记录。"""
    swanlab_cfg = cfg.get("swanlab", {})
    if not swanlab_cfg.get("enabled", True):
        print("[swanlab] 配置为关闭，跳过训练曲线记录。")
        return None

    try:
        from swanlab.integration.transformers import SwanLabCallback
    except ImportError:
        print("[swanlab] 未安装 swanlab，跳过训练曲线记录（pip install swanlab 后可启用）。")
        return None

    project = swanlab_cfg.get("project", "demo3-bc2gm-ner")
    print(f"[swanlab] 项目: {project} | 实验名: {run_name}")
    print("[swanlab] 提醒：提交前请到 SwanLab 网页确认该项目已开放访问权限。")
    return SwanLabCallback(project=project, experiment_name=run_name)
