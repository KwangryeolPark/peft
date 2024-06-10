import math
import torch
from typing import Any, List, Optional

from torch import nn
from peft.tuners.lora import LoraLayer
    
class SoraLayer(LoraLayer):
    adapter_layer_names = ("lora_A", "lora_B", "lora_G", "lora_embedding_A", "lora_embedding_B")
    
    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_G = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
        self.lora_G[adapter_name] = nn.Parameter(torch.randn(r, 1))
        self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])

class SoraLinear(nn.Module, SoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        SoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_G = self.lora_G[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x = x.to(lora_A.dtype)
            result += (dropout(x) @ (lora_A * lora_G).T @ lora_B.T) * scaling

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sora." + rep