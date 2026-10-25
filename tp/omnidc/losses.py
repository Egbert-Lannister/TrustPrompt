"""Fine-tuning losses (docs/plan_omnidc.md sec. 5, 16).

  OMNI-DC terms   src/training_scripts/train_full.sh without SeqLaplace (its head is not in the v1.1 weights) and
                  without ConfInput, which is handled here: 1 SeqGradL1 + 1 SeqL1 + 2 GradMatchingScale, OMNI-DC's
                  own SequentialLoss.
  ConfInput       f0: OMNI-DC's own (src/loss/submodule/confinputloss.py: a cell is noisy iff some reading exceeds
                  the GT by 0.1 in median-normalised units, i.e. one-sided), re-stated here because its
                  torch.abs on a boolean tensor fails in current torch (the abs is a no-op on 0 / 1, so the
                  arithmetic is unchanged; tests/test_omnidc_train.py). f1 / f2: two-sided, from the labels:
                  a cell is noisy iff some reading in it is measured too far or too near (|P - GT| / GT > tau).
                  Same BCE on OMNI-DC's input-confidence map, same valid cells (cells with a reading).
                  soft (P1 v2, docs/plan_omnidc.md sec. 21): the same two-sided labels, but the target of a cell is
                  the share of its readings that are correct, so one wrong reading among 16 gives 15/16, not 0.
  gate            f2: cross-entropy of the gate's clean / far / near logits on readings with GT.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .gate import gate_targets

OMNIDC_TERMS = "1.0*SeqGradL1+1.0*SeqL1+2.0*GradMatchingScale"


def _confidence_bce(conf, noisy, dep):
    """OMNI-DC's ConfInput reduction: BCE against 1 - noisy on cells with a reading, mean per image, summed."""
    valid = F.max_pool2d((dep > 1e-4).float(), dep.shape[-1] // conf.shape[-1])
    loss = torch.nan_to_num(F.binary_cross_entropy(conf, 1.0 - noisy, reduction="none") * valid)
    return (loss.sum(dim=(1, 2, 3)) / (valid.sum(dim=(1, 2, 3)) + 1e-8)).sum()


def confinput_original(conf, dep, gt):
    """OMNI-DC's ConfInputLoss.forward without the boolean abs."""
    return _confidence_bce(conf, F.max_pool2d(((dep - gt) > 0.1).float(), dep.shape[-1] // conf.shape[-1]), dep)


def confinput_two_sided(conf, labels, dep):
    """conf [B,1,h,w] (sigmoid), labels [B,H,W] (type_labels), dep [B,1,H,W]."""
    wrong = ((labels == 1) | (labels == 2)).float()[:, None]
    return _confidence_bce(conf, F.max_pool2d(wrong, dep.shape[-1] // conf.shape[-1]), dep)


def confinput_soft(conf, labels, dep):
    """Two-sided labels; cell target = share of correct readings among the cell's readings (valid cells as above)."""
    r = dep.shape[-1] // conf.shape[-1]
    has = (dep > 1e-4).float()
    wrong = ((labels == 1) | (labels == 2)).float()[:, None] * has
    share = F.avg_pool2d(wrong, r) / F.avg_pool2d(has, r).clamp(min=1e-8)
    return _confidence_bce(conf, share.clamp(0, 1), dep)


LABELS = ("original", "two_sided", "soft")


def default_label(variant):
    return "original" if variant == "f0" else "two_sided"


class FineTuneLoss:
    def __init__(self, omnidc_args, variant, gate_weight=1.0, label=None):
        """omnidc_args: the OGNIDC args namespace (its loss fields are overwritten for the OMNI-DC terms).
        label: ConfInput target, original / two_sided / soft (default: original for f0, two_sided otherwise)."""
        from baselines.completion import _sys_path, source_dir
        omnidc_args.loss = OMNIDC_TERMS
        omnidc_args.intermediate_loss_weight = 1.0
        with _sys_path(source_dir("omnidc") / "src"):            # OMNI-DC's own loss modules
            from loss.sequentialloss import SequentialLoss
            self.omnidc = SequentialLoss(omnidc_args)
        self.variant, self.gate_weight = variant, gate_weight
        self.label = label or default_label(variant)
        if self.label not in LABELS:
            raise ValueError(f"unknown ConfInput label {self.label}")

    def cuda(self):
        self.omnidc.cuda(0)
        return self

    def __call__(self, sample, out):
        """Returns (total, dict of named scalar terms)."""
        total_omni, _ = self.omnidc(sample, out)
        terms = dict(omnidc=total_omni.sum())
        if self.label == "original":
            terms["confinput"] = confinput_original(out["confidence_input"], sample["dep"], sample["gt"])
        elif self.label == "soft":
            terms["confinput"] = confinput_soft(out["confidence_input"], sample["labels"], sample["dep"])
        else:
            terms["confinput"] = confinput_two_sided(out["confidence_input"], sample["labels"], sample["dep"])
        if "gate_logits" in out:
            targets = gate_targets(sample["labels"], sample["dep"][:, 0] > 0)
            terms["gate"] = self.gate_weight * F.cross_entropy(out["gate_logits"], targets, ignore_index=-100)
        return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}
