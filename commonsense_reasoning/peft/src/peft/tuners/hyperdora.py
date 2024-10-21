# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

class HypernetworkMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dims):
        super(HypernetworkMLP, self).__init__()
        # output_dims is a list of output dimensions for lora_A, lora_B, and magnitude_weights
        self.mlp_common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_A = nn.Linear(hidden_dim, output_dims[0])
        self.mlp_B = nn.Linear(hidden_dim, output_dims[1])
        self.mlp_C = nn.Linear(hidden_dim, output_dims[2])

        # Initialize the layers' weights to zero
        self.mlp_A.bias.data.zero_()
        self.mlp_B.weight.data.zero_()
        self.mlp_B.bias.data.zero_()
        self.mlp_C.weight.data.zero_()
        self.mlp_C.bias.data.zero_()

    def forward(self, x):
        common = self.mlp_common(x)
        output_A = self.mlp_A(common)
        output_B = self.mlp_B(common)
        output_C = self.mlp_C(common)
        return output_A, output_B, output_C

@dataclass
class HyperDoraConfig(PeftConfig):
    """
    Configuration class to store the configuration of a HyperLora model.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with HyperLora."
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    hypernetwork_input_dim: int = field(default=128, metadata={"help": "Input dimension for hypernetwork MLP"})
    hypernetwork_hidden_dim: int = field(default=32, metadata={"help": "Hidden dimension for hypernetwork MLP"})
    input_std: float = field(default=0.01, metadata={"help": "Hypernet input std"})

    def __post_init__(self):
        self.peft_type = PeftType.HYPERDORA


class HyperDoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (HyperLora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`HyperLoraConfig`]): The configuration of the HyperLora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_hyper_dora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use HyperLora with 8-bit quantization, please install the `bitsandbytes` package."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "hypernetwork_input_dim": self.peft_config.hypernetwork_input_dim,
            "hypernetwork_hidden_dim": self.peft_config.hypernetwork_hidden_dim,
            "input_std": self.peft_config.input_std,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    new_module = HyperDoraLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear):
                    new_module = HyperDoraLinear(target.in_features, target.out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        with torch.no_grad():
            magnitude = (torch.linalg.norm(new_module.weight.detach(),dim=1)).unsqueeze(1).detach()
            new_module.weight_m_wdecomp.weight.copy_(magnitude)


        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "hyper_dora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped model."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, HyperDoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


def mark_only_hyper_dora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "hypernetwork" not in n and  "input_vector" not in n and "rescale" not in n :
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, HyperDoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class HyperDoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        hypernetwork: Optional[nn.Module] = None,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.hypernetwork = hypernetwork  # Hypernetwork MLP
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class HyperDoraLinear(nn.Linear, HyperDoraLayer):
    """
    Implements HyperDora using a Linear layer with hypernetwork MLP for parameter generation and Dora decomposition.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        hypernetwork_input_dim: int = 128,
        hypernetwork_hidden_dim: int = 256,
        input_std: float = 0.01,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.hypernetwork = HypernetworkMLP(hypernetwork_input_dim, hypernetwork_hidden_dim, [r * in_features, r * out_features, out_features]  )
        input_vector = torch.empty(1, hypernetwork_input_dim).normal_(mean=0.0, std=input_std)
        #input_vector = torch.empty(1, hypernetwork_input_dim).normal_(mean=0.0, std=1.0)
        self.input_vector = nn.Parameter(input_vector)
        HyperDoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, hypernetwork=self.hypernetwork)
        self.fan_in_fan_out = fan_in_fan_out
        self.dora_simple = True  # Simplified to save GPU memory, can be turned off if needed
        if r > 0:
            self.weight.requires_grad = False
            self.scaling = self.lora_alpha / self.r
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Dora Decomposition: Separate magnitude and direction
        self.weight_m_wdecomp = nn.Linear(1, out_features, bias=False)  # Magnitude component

    def generate_lora_parameters(self):
        if self.hypernetwork is not None and self.input_vector is not None:
            output_A, output_B, output_C = self.hypernetwork(self.input_vector)
            lora_A_size = (self.in_features, self.r)
            lora_B_size = (self.r, self.out_features)
            self.lora_A = output_A.reshape(lora_A_size)
            self.lora_B = output_B.reshape(lora_B_size)
            self.magnitude_weights = output_C.reshape(-1)

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
        self.generate_lora_parameters()


        # Apply DoRA adaptation if r > 0
        if self.r > 0:

            new_weight_v = self.weight + (self.lora_B.T @ self.lora_A.T) * self.scaling

            norm_scale = (self.weight_m_wdecomp.weight.view(-1) + self.magnitude_weights) / (torch.linalg.norm(new_weight_v,dim=1)).detach()

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            dropout_x = self.lora_dropout(x)

            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)) )


            result += ( norm_scale * ((self.lora_dropout(x.to(self.lora_A.dtype)) @ self.lora_A) @ self.lora_B)) * self.scaling


        #return result
        return result


