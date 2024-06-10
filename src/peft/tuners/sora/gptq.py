import torch
from .layer import SoraLayer

class SoraQuantLinear(torch.nn.Module, SoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        SoraLayer.__init__(self, base_layer)
        
        self.quant_linear_module = base_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.quant_linear_module(x)
        
        if self.disable_adapters:
            return result
        
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_G = self.lora_G[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
            
            output = (dropout(x) @ (lora_A * lora_G).T @ lora_B.T) * scaling
            
            if requires_conversion:
                output = output.to(expected_dtype)
            result += output
        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sora." + rep