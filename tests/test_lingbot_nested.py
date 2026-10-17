import sys

import pytest

from baselines.completion import patch_nested_blocks, source_dir

torch = pytest.importorskip("torch")
SRC = source_dir("lingbot")
if not (SRC / "mdm").is_dir():
    pytest.skip("LingBot source not checked out", allow_module_level=True)
sys.path.insert(0, str(SRC))
from mdm.model.dinov2_rgbd.layers import block as lb          # noqa: E402
from mdm.model.dinov2_rgbd.layers import MemEffAttention       # noqa: E402


def block_diagonal_reference(blk, xs):
    """What xFormers computes: one concatenated sequence, attention restricted to each sample's tokens."""
    x = torch.cat(xs, 1); n = x.shape[1]; C = x.shape[2]; H = blk.attn.num_heads
    seg = torch.cat([torch.full((t.shape[1],), i) for i, t in enumerate(xs)])
    mask = seg[:, None] == seg[None, :]
    h = blk.norm1(x)
    q, k, v = blk.attn.qkv(h).reshape(1, n, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    o = blk.attn.proj(o.permute(0, 2, 1, 3).reshape(1, n, C))
    x = x + blk.ls1(o)
    x = x + blk.ls2(blk.mlp(blk.norm2(x)))
    return list(torch.split(x, [t.shape[1] for t in xs], dim=1))


def test_patched_nested_block_equals_block_diagonal_attention():
    torch.manual_seed(0)
    blk = lb.NestedTensorBlock(dim=64, num_heads=4, attn_class=MemEffAttention, init_values=0.5).eval()
    xs = [torch.randn(1, 7, 64), torch.randn(1, 11, 64)]
    patch_nested_blocks(lb)
    with torch.no_grad():
        got = blk(xs); want = block_diagonal_reference(blk, xs)
        single = blk(torch.cat(xs[:1], 1))
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=1e-5)
    assert torch.allclose(got[0], single, atol=1e-6)
    blk.train()
    with pytest.raises(RuntimeError):
        blk(xs)
