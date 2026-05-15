import datetime


def log_info(msg: str) -> None:
    """统一的中文日志打印，带时间戳。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"【INFO】【{now}】{msg}")


def log_bt(msg: str) -> None:
    """BT 相关的日志打印，突出桶分裂与采样细节。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO][BT][{now}] {msg}")
