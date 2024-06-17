import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)
from peft.utils.integrations import gather_params_ctx

from .gptq import SoraQuantLinear
from .layer import SoraLayer, SoraLinear

class SoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
    
    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if (kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]) and not is_bnb_available():
            raise ImportError(
                "To use AdaLora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        # If it is not an AdaLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, SoraLayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import SoraLinear8bitLt
        if is_bnb_4bit_available():
            from .bnb import SoraLinear4bit

        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = SoraLinear8bitLt(target, adapter_name, **kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = SoraLinear4bit(target, adapter_name, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = SoraQuantLinear(target, adapter_name, **kwargs)
        else:
            if isinstance(target_base_layer, torch.nn.Linear):
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target_base_layer, Conv1D):
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = SoraLinear(target, adapter_name, **kwargs)

        return new_module

    # @staticmethod
    # def _prepare_adapter_config(peft_config, model_config):
    #     if peft_config.target_modules is None:
    #         if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
    #             raise ValueError("Please specify `target_modules` in `peft_config`")
    #         peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
    #             model_config["model_type"]
    #         ]
    #     return peft_config

    # def __getattr__(self, name: str):
    #     """Forward missing attributes to the wrapped module."""
    #     try:
    #         return super().__getattr__(name)  # defer to nn.Module's logic
    #     except AttributeError:
    #         return getattr(self.model, name)

    # def forward(self, *args, **kwargs):
    #     pass
