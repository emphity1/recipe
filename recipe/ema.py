"""EMA / tail-average of model weights for the Ralph training loop.

Saving an exponential-moving-average (EMA) of the weights over the WSD low-LR
decay tail — instead of the single raw final-step weights — flattens the
minimum and improves generalization. This is disproportionately valuable on
Ralph: training is English-only fineweb-edu but the sealed hidden eval is a
multi-stream pool (English/code/math/multilingual), so a flatter minimum
transfers better out-of-distribution. It also cuts run-to-run eval variance.

Safety / correctness invariants (all verified against the op4 loader):
  * op4 rebuilds the CANONICAL RalphBase and does a STRICT load_state_dict
    (validator/eval_in_workdir.py, validator/sandbox_eval.py). Averaging must
    therefore keep the state_dict KEYS and SHAPES identical to RalphBase and
    only touch VALUES. This helper only ever averages values.
  * rope_cache is registered persistent=False in model/_v7lite.py, so it is
    absent from state_dict() and never averaged (only learnable params are).
  * The shadow is (re)initialised to the LIVE weights at the FIRST update
    (>= start_step), so pre-decay / random-init weights never leak into the
    tail soup.
  * Only floating-point tensors are averaged; any non-float entry (none exist
    in RalphBase today, but this is defensive) tracks the live value verbatim.
  * update() is a deterministic function of (weights, decay, step): the Stage-5
    audit re-run of our patched train.py reproduces the same soup.
"""

from __future__ import annotations

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, start_step: int = 0):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.n_updates = 0
        # float32 shadow of the persistent state (params + persistent buffers).
        # Non-float entries are kept in their native dtype (defensive; RalphBase
        # has none today).
        self._shadow = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self._shadow[k] = v.detach().float().clone()
            else:
                self._shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        if self.decay <= 0.0 or step < self.start_step:
            return
        sd = model.state_dict()
        first = self.n_updates == 0
        for k, s in self._shadow.items():
            v = sd[k]
            if not torch.is_floating_point(v):
                s.copy_(v)
                continue
            if first:
                # Initialise the average at the moment averaging begins so the
                # pre-decay weights never pollute the tail soup.
                s.copy_(v.float())
            else:
                s.mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)
        self.n_updates += 1

    def state_dict(self, model: torch.nn.Module) -> dict:
        """Averaged weights cast back to each param's original dtype, keyed
        identically to model.state_dict(). Falls back to the live weights if no
        averaging happened (start_step never reached) so a mis-set window can
        never save a worse-than-final checkpoint."""
        live = model.state_dict()
        if self.n_updates == 0:
            return live
        out = {}
        for k, v in live.items():
            if torch.is_floating_point(v):
                out[k] = self._shadow[k].to(v.dtype)
            else:
                out[k] = v
        return out
