import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model._v4skip import RalphBase, RalphConfig


class ReadoutBase(RalphBase):
    """RalphBase + per-vocab readout-calibration head (logit_scale + readout_gain/bias).
    Reproduces the eff5h-readout arch. 0-init => exp(0)=1,+0 => identity at step 0."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.readout_gain = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.readout_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assert idx.shape[-1] <= self.cfg.max_seq_len
        x = self.tok_embed(idx)
        if self.unet_skip:
            n = len(self.blocks); half = n // 2; enc = []
            for i, block in enumerate(self.blocks):
                if i < half:
                    x = block(x, self.rope_cache); enc.append(x)
                else:
                    x = x + self.skip_gate[i - half] * enc[n - 1 - i]
                    x = block(x, self.rope_cache)
        else:
            for block in self.blocks:
                x = block(x, self.rope_cache)
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_embed.weight)
        else:
            logits = self.lm_head(x)
        logits = logits * torch.exp(self.readout_gain) + self.readout_bias
        logits = logits * torch.exp(self.logit_scale)
        cap = getattr(self.cfg, "logit_softcap", 0.0)
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
            )
        return logits, loss
