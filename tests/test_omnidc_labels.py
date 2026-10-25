import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from tp.omnidc.losses import _confidence_bce, confinput_soft, confinput_two_sided, default_label  # noqa: E402


def test_soft_target_is_share_of_correct_readings():
    dep = torch.ones(1, 1, 8, 8); dep[0, 0, 4:, 4:] = 0            # bottom-right cell has no reading
    lab = torch.zeros(1, 8, 8, dtype=torch.long)
    lab[0, 0, 0] = 1                                                # top-left cell: 1 of 16 wrong
    lab[0, :4, 4:] = 2                                              # top-right cell: all wrong
    conf = torch.full((1, 1, 2, 2), 0.7)
    noisy = torch.tensor([[[[1 / 16, 1.0], [0.0, 0.0]]]])
    assert torch.allclose(confinput_soft(conf, lab, dep), _confidence_bce(conf, noisy, dep))
    # the two-sided max label calls the top-left cell fully noisy
    assert torch.allclose(confinput_two_sided(conf, lab, dep), _confidence_bce(conf, torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]), dep))
    # a confident head is penalised much less by the soft target on a mostly correct cell
    hi = torch.full_like(conf, 0.95)
    one = torch.zeros_like(lab); one[0, 0, 0] = 1
    assert confinput_soft(hi, one, torch.ones(1, 1, 8, 8)) < 0.2 * confinput_two_sided(hi, one, torch.ones(1, 1, 8, 8))


def test_default_labels_keep_p1_behaviour():
    assert default_label("f0") == "original" and default_label("f1") == "two_sided" and default_label("f2") == "two_sided"
