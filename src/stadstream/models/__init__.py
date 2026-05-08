from .dpwit import DPWiTDetector
from .matr_signal import MATRSignalBaseline
from .moad_signal import MOADSignalBaseline
from .slimstad import SlimSTADDetector
from .stadstream import STADStreamNet


def build_model(name: str, **kwargs):
    name = name.lower()
    if name in {"dpwit", "wifitad"}:
        return DPWiTDetector(**kwargs)
    if name == "slimstad":
        return SlimSTADDetector(**kwargs)
    if name in {"matr", "matr_signal", "ontal"}:
        return MATRSignalBaseline(**kwargs)
    if name in {"moad", "moad_signal"}:
        return MOADSignalBaseline(**kwargs)
    if name in {"stadstream", "stadstream_v1"}:
        return STADStreamNet(**kwargs)
    raise ValueError(f"Unknown model: {name}")
