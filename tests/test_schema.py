import numpy as np
import pytest
from scripts.make_synthetic import make_sample
from tp.data.schema import save_sample,load_sample,validate_scene_splits


def test_roundtrip_preserves_raw_holes_and_units(tmp_path):
    s=make_sample()
    s.prompt[0,0]=np.nan
    s.prompt_valid[0,0]=False
    p=tmp_path/"sample.npz"
    save_sample(s,p)
    r=load_sample(p)
    np.testing.assert_array_equal(s.prompt,r.prompt)
    assert np.isnan(r.prompt[0,0]) and not r.prompt_valid[0,0]
    assert r.metadata["depth_unit"]=="m"
    assert r.prompt_conf is None
    with pytest.raises(FileExistsError):
        save_sample(s,p)


def test_invalid_masks_units_and_material_fail():
    s=make_sample()
    s.prompt_valid[:]=True
    with pytest.raises(ValueError,match="nonfinite/nonpositive"):
        s.validate()
    s=make_sample();s.metadata["depth_unit"]="mm"
    with pytest.raises(ValueError,match="depth_unit"):
        s.validate()
    s=make_sample();s.material[0,0]=255
    with pytest.raises(ValueError,match="material"):
        s.validate()


def test_scene_split_leakage_rejected():
    rows=[dict(dataset="d",scene="s",frame="1",split="train"),dict(dataset="d",scene="s",frame="2",split="test")]
    with pytest.raises(ValueError,match="scene leakage"):
        validate_scene_splits(rows)


def test_multiple_sensors_share_scene_split_but_not_sample_identity():
    rows=[dict(dataset="d",scene="s",frame="1",sensor=s,split="train") for s in ("d435","l515","helios")]
    validate_scene_splits(rows)
    rows[1]["split"]="test"
    with pytest.raises(ValueError,match="scene leakage"):
        validate_scene_splits(rows)
