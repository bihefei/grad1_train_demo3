
from transformers import TrainerCallback


class _SafeSwanLabCallback(TrainerCallback):
    def __init__(self, inner):
        self._inner = inner

    def _safe(self, method_name, *args, **kwargs):
        try:
            return getattr(self._inner, method_name)(*args, **kwargs)
        except Exception as e:
            print(f"[swanlab] 回调 {method_name} 执行出错，已忽略（不影响训练）: {e}")
            return None

    def on_init(self, *args, **kwargs):           return self._safe("on_init", *args, **kwargs)
    def on_train_begin(self, *args, **kwargs):    return self._safe("on_train_begin", *args, **kwargs)
    def on_train_end(self, *args, **kwargs):      return self._safe("on_train_end", *args, **kwargs)
    def on_epoch_begin(self, *args, **kwargs):    return self._safe("on_epoch_begin", *args, **kwargs)
    def on_epoch_end(self, *args, **kwargs):    return self._safe("on_epoch_end", *args, **kwargs)
    def on_step_begin(self, *args, **kwargs):     return self._safe("on_step_begin", *args, **kwargs)
    def on_step_end(self, *args, **kwargs):       return self._safe("on_step_end", *args, **kwargs)
    def on_substep_end(self, *args, **kwargs):    return self._safe("on_substep_end", *args, **kwargs)
    def on_save(self, *args, **kwargs):           return self._safe("on_save", *args, **kwargs)
    def on_evaluate(self, *args, **kwargs):       return self._safe("on_evaluate", *args, **kwargs)
    def on_predict(self, *args, **kwargs):        return self._safe("on_predict", *args, **kwargs)
    def on_log(self, *args, **kwargs):            return self._safe("on_log", *args, **kwargs)


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

    try:
        project = swanlab_cfg.get("project", "demo3-bc2gm-ner")
        inner = SwanLabCallback(project=project, experiment_name=run_name)
    except Exception as e:
        # 常见于未登录 swanlab / 网络不通 / 离线模式本地看板崩溃
        print(f"[swanlab] 初始化失败，跳过训练曲线记录（不影响训练）: {e}")
        return None

    print(f"[swanlab] 项目: {project} | 实验名: {run_name}")
    print("[swanlab] 若运行环境无法使用 swanlab（未登录/无网络），"
          "把 config 中 swanlab.enabled 改为 false 即可完全跳过，不影响训练。")
    return _SafeSwanLabCallback(inner)
