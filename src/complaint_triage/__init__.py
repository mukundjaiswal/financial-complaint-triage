"""Financial complaint triage.

Routes free-text consumer complaints into product categories and summarizes
them, using an open-weight instruction model with no fine-tuning.

The finding the project exists to demonstrate: on the same model and the same
prompt, measured Micro-F1 was **0.667** on raw generations and **0.933** after
output normalization. The model was already choosing correctly; the scoring
layer could not read what it said.

Registration of this project's metrics is an explicit call, not an import side
effect: see :func:`complaint_triage.eval_metrics.register_task_metrics`.
"""

from complaint_triage.eval_metrics import register_task_metrics
from complaint_triage.postprocess import extract_category, normalize
from complaint_triage.settings import Settings, get_settings

__all__ = [
    "Settings",
    "__version__",
    "extract_category",
    "get_settings",
    "normalize",
    "register_task_metrics",
]
__version__ = "0.1.0"
