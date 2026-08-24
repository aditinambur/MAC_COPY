import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapping an existing nn.Linear layer.

    Given a frozen base linear layer W0 in R^{out_features x in_features}, computes:
        output = W0(x) + (alpha / r) * (x @ A.T @ B.T)
    where:
        A in R^{r x in_features} is initialized with Kaiming uniform.
        B in R^{out_features x r} is initialized to zeros, so initially Delta W = 0
        (exact identity behavior with respect to the pre-trained model).
    """

    def __init__(self, base_linear: nn.Linear, r: int = 4, lora_alpha: float = 8.0, enabled: bool = True):
        super(LoRALinear, self).__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0
        self.enabled = enabled
        self.merged = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.reset_parameters()
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_linear(x)
        if self.enabled and self.r > 0 and not self.merged:
            lora_term = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            result = result + lora_term
        return result

    def merge(self):
        """Merge LoRA weights into base_linear.weight for zero-overhead inference."""
        if self.r > 0 and not self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.base_linear.weight.data += delta_w
                self.merged = True

    def unmerge(self):
        """Subtract LoRA weights from base_linear.weight."""
        if self.r > 0 and self.merged:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.base_linear.weight.data -= delta_w
                self.merged = False


def inject_lora_to_actor(actor: nn.Module, r: int = 4, lora_alpha: float = 8.0) -> list:
    """
    Injects LoRALinear into the dense feedforward and action projection layers of an R_Actor:
      - base.mlp.fc1[0]
      - base.mlp.fc2[i][0] for each layer i
      - act.action_out.linear (or act.action_outs[i].linear)
    Returns a list of created LoRALinear modules.
    """
    lora_modules = []

    # 1. Base MLP layers
    if hasattr(actor, 'base') and hasattr(actor.base, 'mlp'):
        mlp = actor.base.mlp
        if hasattr(mlp, 'fc1') and len(mlp.fc1) > 0 and isinstance(mlp.fc1[0], nn.Linear):
            lora_layer = LoRALinear(mlp.fc1[0], r=r, lora_alpha=lora_alpha)
            mlp.fc1[0] = lora_layer
            lora_modules.append(lora_layer)

        if hasattr(mlp, 'fc2'):
            for i in range(len(mlp.fc2)):
                if len(mlp.fc2[i]) > 0 and isinstance(mlp.fc2[i][0], nn.Linear):
                    lora_layer = LoRALinear(mlp.fc2[i][0], r=r, lora_alpha=lora_alpha)
                    mlp.fc2[i][0] = lora_layer
                    lora_modules.append(lora_layer)

    # 2. Action output layers
    if hasattr(actor, 'act'):
        act = actor.act
        if hasattr(act, 'action_out') and hasattr(act.action_out, 'linear') and isinstance(act.action_out.linear, nn.Linear):
            lora_layer = LoRALinear(act.action_out.linear, r=r, lora_alpha=lora_alpha)
            act.action_out.linear = lora_layer
            lora_modules.append(lora_layer)
        elif hasattr(act, 'action_outs'):
            for i in range(len(act.action_outs)):
                if hasattr(act.action_outs[i], 'linear') and isinstance(act.action_outs[i].linear, nn.Linear):
                    lora_layer = LoRALinear(act.action_outs[i].linear, r=r, lora_alpha=lora_alpha)
                    act.action_outs[i].linear = lora_layer
                    lora_modules.append(lora_layer)

    return lora_modules


def eject_lora_from_actor(actor: nn.Module):
    """
    Restores the original nn.Linear layers in an R_Actor, removing LoRA wrappers.
    """
    if hasattr(actor, 'base') and hasattr(actor.base, 'mlp'):
        mlp = actor.base.mlp
        if hasattr(mlp, 'fc1') and len(mlp.fc1) > 0 and isinstance(mlp.fc1[0], LoRALinear):
            mlp.fc1[0] = mlp.fc1[0].base_linear
        if hasattr(mlp, 'fc2'):
            for i in range(len(mlp.fc2)):
                if len(mlp.fc2[i]) > 0 and isinstance(mlp.fc2[i][0], LoRALinear):
                    mlp.fc2[i][0] = mlp.fc2[i][0].base_linear

    if hasattr(actor, 'act'):
        act = actor.act
        if hasattr(act, 'action_out') and hasattr(act.action_out, 'linear') and isinstance(act.action_out.linear, LoRALinear):
            act.action_out.linear = act.action_out.linear.base_linear
        elif hasattr(act, 'action_outs'):
            for i in range(len(act.action_outs)):
                if hasattr(act.action_outs[i], 'linear') and isinstance(act.action_outs[i].linear, LoRALinear):
                    act.action_outs[i].linear = act.action_outs[i].linear.base_linear


def set_actor_lora_trainable(actor: nn.Module, trainable_comm: bool = True):
    """
    Freezes all base parameters in the actor and sets requires_grad=True strictly for:
      - All lora_A and lora_B parameters in injected LoRALinear modules.
      - (Optional) token_embedding and message_head for communication pathway repair.
    """
    # Freeze all actor parameters first
    for p in actor.parameters():
        p.requires_grad = False

    # Enable LoRA parameters
    for m in actor.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            if m.lora_A is not None:
                m.lora_A.requires_grad = True
            if m.lora_B is not None:
                m.lora_B.requires_grad = True

    # Enable communication layers if requested
    if trainable_comm:
        if hasattr(actor, 'token_embedding') and actor.token_embedding is not None:
            for p in actor.token_embedding.parameters():
                p.requires_grad = True
        if hasattr(actor, 'message_head') and actor.message_head is not None:
            for p in actor.message_head.parameters():
                p.requires_grad = True
