from dataclasses import dataclass

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType

@dataclass
class SoraConfig(LoraConfig):
    def __post_init__(self):
        self.peft_type = PeftType.SORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )