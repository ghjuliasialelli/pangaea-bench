"""LoRA (low-rank adaptation) for the encoders.

LoRA is injected as a weight parametrization (W -> W + B @ A * alpha / r) rather than by
wrapping nn.Linear modules, because several encoders read projection weights directly
(e.g. ``F.linear(x, self.qkv.weight)`` in SSL4EO-Data2Vec and torchvision's Swin-V2 in
SatlasNet) and nn.MultiheadAttention (RemoteCLIP, DOFA) uses a raw ``in_proj_weight``.
A module wrapper is silently bypassed in those cases; a parametrization is not.

B is zero-initialised, so the adapted encoder is exactly the pretrained one at step 0.
"""

import re
from logging import Logger

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

ATTENTION_CLASS_RE = re.compile(r"attn|attention", re.IGNORECASE)


class LoRAParametrization(nn.Module):
    """Adds a trainable low-rank update B @ A * (alpha / r) to a 2D weight."""

    def __init__(self, out_features: int, in_features: int, r: int, alpha: float):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = alpha / r

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.lora_B @ self.lora_A).to(weight.dtype) * self.scale


def find_lora_targets(
    encoder: nn.Module, target_modules: list[str] | None = None
) -> list[tuple[str, str]]:
    """Find the attention projection weights to adapt.

    Targets are the nn.Linear children of every module whose class name contains
    "attn"/"attention", plus the input/output projections of every nn.MultiheadAttention.

    Args:
        encoder (nn.Module): the encoder.
        target_modules (list[str] | None): if given, keep only targets whose module name
            (last component, e.g. "qkv", "proj", "in_proj") is in this list.

    Returns:
        list[tuple[str, str]]: (module path, parameter name) pairs.
    """
    targets = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                targets.append((name, "in_proj_weight"))
            targets.append((f"{name}.out_proj", "weight"))
        elif ATTENTION_CLASS_RE.search(type(module).__name__):
            for child_name, child in module.named_children():
                # exact type: MHA's out_proj (a Linear subclass) is handled above
                if type(child) is nn.Linear:
                    targets.append((f"{name}.{child_name}" if name else child_name, "weight"))

    if target_modules is not None:
        wanted = set(target_modules)
        targets = [
            (m, p)
            for m, p in targets
            if (p[: -len("_weight")] if p == "in_proj_weight" else m.rsplit(".", 1)[-1]) in wanted
        ]
    return targets


def apply_lora(
    encoder: nn.Module,
    r: int,
    alpha: float,
    target_modules: list[str] | None,
    logger: Logger,
) -> None:
    """Inject LoRA into the encoder's attention projections and freeze everything else.

    Must be called after the pretrained weights are loaded (the parametrization renames
    the adapted weights to ``...parametrizations.<name>.original``), and before any
    checkpoint of a LoRA run is loaded.

    Args:
        encoder (nn.Module): the encoder, modified in place.
        r (int): rank of the update.
        alpha (float): scaling numerator; the update is scaled by alpha / r.
        target_modules (list[str] | None): see find_lora_targets.
        logger (Logger): logger.
    """
    targets = find_lora_targets(encoder, target_modules)
    if not targets:
        raise ValueError(
            f"LoRA found no attention projections to adapt in {type(encoder).__name__} "
            f"(target_modules={target_modules}). Convolutional encoders are not supported."
        )

    for module_name, param_name in targets:
        module = encoder.get_submodule(module_name)
        weight = getattr(module, param_name)
        out_features, in_features = weight.shape
        parametrize.register_parametrization(
            module,
            param_name,
            LoRAParametrization(out_features, in_features, r, alpha).to(weight.device),
        )

    for name, param in encoder.named_parameters():
        param.requires_grad = ".lora_A" in name or ".lora_B" in name

    n_lora = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    kinds = sorted({p if p != "weight" else m.rsplit(".", 1)[-1] for m, p in targets})
    logger.info(
        f"LoRA (r={r}, alpha={alpha}): adapted {len(targets)} weights {kinds}; "
        f"{n_lora:,} trainable of {n_total:,} encoder parameters ({100 * n_lora / n_total:.2f}%)."
    )


def check_lora_gradients(model: nn.Module, logger: Logger) -> list[str]:
    """Report LoRA adapters that received no gradient, i.e. that will never train.

    ``apply_lora`` can only verify that attention projections *exist*; it cannot know
    which of them the forward pass actually reaches. An encoder whose ``output_layers``
    stop short of its full depth -- CROMA-large is 24 layers deep and its configs ask for
    [3, 5, 7, 11] -- leaves every adapter past that point pinned at its zero-initialised
    value. The run trains, converges and reports a number for an encoder that was only
    partly adapted, and DDP's ``find_unused_parameters=True`` absorbs it silently. The
    log line ``apply_lora`` emits is no help: it counts adapters installed, not adapters
    reached.

    A ``grad`` of None is unambiguous here: an adapter sits on a weight that is read
    whenever its module runs, so None means the module never ran, or its output never
    reached the loss. Warn rather than raise -- adapting only the first N layers is a
    legitimate thing to configure, it just has to be deliberate.

    Args:
        model (nn.Module): the model (or encoder), after at least one backward pass.
        logger (Logger): logger.

    Returns:
        list[str]: names of the adapter tensors without a gradient; empty if the model
            has no LoRA adapters at all, so this is safe to call on every run.
    """
    adapters = [
        (name, param)
        for name, param in model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    ]
    if not adapters:
        return []

    dead = [name for name, param in adapters if param.grad is None]
    if dead:
        modules = sorted({name.rsplit(".parametrizations.", 1)[0] for name in dead})
        logger.warning(
            f"LoRA: {len(dead)} of {len(adapters)} adapter tensors got no gradient in the "
            f"first backward pass. Those layers are NOT being adapted and will keep their "
            f"zero-initialised update for the whole run -- the encoder is partly frozen. "
            f"{len(modules)} module(s) affected: {modules}"
        )
    else:
        logger.info(f"LoRA: all {len(adapters)} adapter tensors received gradients.")
    return dead
