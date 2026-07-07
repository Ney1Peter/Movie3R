import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with a frozen-compatible low-rank residual branch.

    The base weight/bias keep the original ``weight`` and ``bias`` parameter
    names so existing Human3R checkpoints still load into wrapped layers.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, self.out_features, bias=False)
        self.lora_enabled = True
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if not self.lora_enabled:
            return base
        lora = self.lora_up(self.lora_down(self.lora_dropout(x)))
        return base + lora.to(dtype=base.dtype) * self.scaling

    def mark_only_lora_trainable(self, trainable: bool = True) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.lora_down.weight.requires_grad = trainable
        self.lora_up.weight.requires_grad = trainable

    def lora_l2(self) -> torch.Tensor:
        return 0.5 * (
            self.lora_down.weight.pow(2).mean() + self.lora_up.weight.pow(2).mean()
        )


def iter_lora_layers(module: nn.Module) -> Iterable[LoRALinear]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def inject_lora_to_linear_modules(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    prefix: str = "",
) -> List[str]:
    """Recursively replace nn.Linear children with LoRALinear wrappers."""

    replaced: List[str] = []
    for name, child in list(module.named_children()):
        child_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(child_name)
            continue
        replaced.extend(
            inject_lora_to_linear_modules(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                prefix=child_name,
            )
        )
    return replaced


def mark_lora_trainable(module: nn.Module, trainable: bool = True) -> int:
    count = 0
    for layer in iter_lora_layers(module):
        layer.mark_only_lora_trainable(trainable=trainable)
        count += 1
    return count


def lora_parameter_l2(module: nn.Module):
    values = [layer.lora_l2() for layer in iter_lora_layers(module)]
    if not values:
        return None
    return torch.stack(values).mean()


def set_lora_enabled(module: nn.Module, enabled: bool = True) -> int:
    count = 0
    for layer in iter_lora_layers(module):
        layer.lora_enabled = bool(enabled)
        count += 1
    return count


def count_lora_parameters(module: nn.Module) -> Dict[str, int]:
    trainable = 0
    total = 0
    layers = 0
    for layer in iter_lora_layers(module):
        layers += 1
        for name, param in layer.named_parameters():
            if not name.startswith("lora_"):
                continue
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return {"layers": layers, "lora_params": total, "trainable_lora_params": trainable}
