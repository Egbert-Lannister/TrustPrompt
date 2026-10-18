import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from baselines.completion import source_dir
from baselines.guidance import Guide, complete, guidance_loss


def maps(seed=0, h=6, w=7):
    g = torch.Generator().manual_seed(seed)
    d = torch.rand(1, 1, h, w, generator=g) * 4 + 0.5
    p = torch.rand(1, 1, h, w, generator=g) * 4 + 0.5
    valid = torch.rand(1, 1, h, w, generator=g) > 0.3
    return d, p, valid


def test_loss_without_weights_is_upstream_l1_plus_l2():
    d, p, valid = maps()
    up = torch.nn.functional.l1_loss(d[valid], p[valid]) + torch.nn.functional.mse_loss(d[valid], p[valid])
    assert guidance_loss(d, p, valid) == up
    ones = torch.ones_like(d)
    assert torch.allclose(guidance_loss(d, p, valid, weight=ones), up, rtol=1e-6)


def test_hinge_only_penalises_predictions_behind_the_reading():
    d, p, valid = maps(1)
    zero, one = torch.zeros_like(d), torch.ones_like(d)
    front = torch.minimum(d, p - 0.1)                       # every prediction in front of the reading
    assert guidance_loss(front, p, valid, weight=zero, hinge=one) == 0
    behind = p + 0.5
    n = int(valid.sum())
    assert torch.allclose(guidance_loss(behind, p, valid, weight=zero, hinge=one), torch.tensor((0.5 + 0.25) * n / n))
    # equality + hinge on the same pixel counts the far side twice, the near side once
    assert torch.allclose(guidance_loss(behind, p, valid, weight=one, hinge=one), torch.tensor(1.5))


def test_guide_check_rejects_bad_maps():
    valid = np.ones((4, 5), bool); valid[0] = False
    Guide(weight=np.ones((4, 5), np.float32), anchor=valid).check(valid)
    for bad in (dict(weight=np.ones((4, 5))),                       # float64
                dict(hinge=-np.ones((4, 5), np.float32)),
                dict(anchor=np.ones((4, 5), bool)),                 # anchor outside the valid pixels
                dict(anchor=np.zeros((4, 5), bool)),
                dict(weight=np.ones((3, 5), np.float32))):
        with pytest.raises(ValueError):
            Guide(**bad).check(valid)


@pytest.fixture(scope="module")
def tiny_pipe():
    """A random-weight Marigold pipeline small enough for the CPU (structure as prs-eth/marigold-depth-v1-0)."""
    if not (source_dir("marigold_dc") / "marigold_dc.py").is_file():
        pytest.skip("Marigold-DC checkout missing")
    import sys
    sys.path.insert(0, str(source_dir("marigold_dc")))
    try:
        from marigold_dc import MarigoldDepthCompletionPipeline
    finally:
        sys.path.remove(str(source_dir("marigold_dc")))
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    torch.manual_seed(0)
    unet = UNet2DConditionModel(sample_size=8, in_channels=8, out_channels=4, layers_per_block=1,
                                block_out_channels=(8, 16), norm_num_groups=4, cross_attention_dim=8,
                                attention_head_dim=2, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"))
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 8),
                        layers_per_block=1, norm_num_groups=4,
                        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2)
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085,
                          beta_end=0.012, clip_sample=False, set_alpha_to_one=False, prediction_type="v_prediction",
                          timestep_spacing="trailing")
    pipe = MarigoldDepthCompletionPipeline(unet=unet, vae=vae, scheduler=sched, text_encoder=None, tokenizer=None,
                                           prediction_type="depth")
    pipe.empty_text_embedding = torch.randn(1, 2, 8)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def scene(seed=0, h=24, w=32):
    from PIL import Image
    rng = np.random.default_rng(seed)
    rgb = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    depth = (rng.random((h, w)) * 3 + 1).astype(np.float32)
    depth[rng.random((h, w)) < 0.4] = 0
    return rgb, depth


def test_equality_guide_reproduces_the_upstream_pipeline(tiny_pipe):
    rgb, depth = scene()
    args = dict(num_inference_steps=4, ensemble_size=2, processing_resolution=32, seed=7)
    up = tiny_pipe(image=rgb, sparse_depth=depth, **args)
    ours = complete(tiny_pipe, rgb, depth, Guide.equality(), **args)
    assert ours.shape == depth.shape and np.array_equal(ours, up)


def test_guides_change_the_result_and_flagged_pixels_respect_the_hinge(tiny_pipe):
    rgb, depth = scene(1)
    valid = depth > 0
    flag = valid.copy(); flag[:, 16:] = False
    far = depth.copy(); far[flag] = 20.0                          # wrong, far readings in the flagged half
    args = dict(num_inference_steps=4, ensemble_size=1, processing_resolution=32, seed=7)
    plain = complete(tiny_pipe, rgb, far, Guide.equality(), **args)
    g = Guide(weight=(valid & ~flag).astype(np.float32), hinge=flag.astype(np.float32), anchor=valid & ~flag)
    guided = complete(tiny_pipe, rgb, far, g, **args)
    assert not np.array_equal(plain, guided) and np.all(np.isfinite(guided))
    # weight 1 everywhere with an all-valid anchor is the same objective as upstream up to float rounding
    same = complete(tiny_pipe, rgb, far, Guide(weight=valid.astype(np.float32), anchor=valid), **args)
    assert np.allclose(same, plain, rtol=1e-4, atol=1e-4)
