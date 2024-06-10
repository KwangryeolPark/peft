from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .config import SoraConfig
from .gptq import SoraQuantLinear
from .layer import SoraLayer, SoraLinear
from .model import SoraModel

__all__ = ["SoraConfig", "SoraQuantLinear", "SoraLayer", "SoraLinear", "SoraModel"]


def __getattr__(name):
    if (name == "SoraLinear8bitLt") and is_bnb_available():
        from .bnb import SoraLinear8bitLt

        return SoraLinear8bitLt

    if (name == "SoraLinear4bit") and is_bnb_4bit_available():
        from .bnb import SoraLinear4bit

        return SoraLinear4bit

    raise AttributeError(f"module {__name__} has no attribute {name}")
