"""BSM、弱桶可靠性和祖先回退入口。"""

from .structural import compute_bucket_gain, compute_bucket_score

__all__ = ["compute_bucket_gain", "compute_bucket_score"]
