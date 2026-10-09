"""Synthetic on-disk fixtures for Mirror3D identity and units regressions.

These are engineering tests, never real dataset results. Arrays retain NYU's
native HDF5 [W=640,H=480] layout to catch crop/transposition/index mistakes.
No external data disk, network, GLARE imports, or real test payload is used.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile

import h5py
import numpy as np
from PIL import Image
import pytest

from tp.data.mirror3d import (
    CROP, SHAPE, REGISTERED_K, Mirror3DNYU, annotation_index,
    read_png, reconstruct_refined,
)


def _png(array, *, palette=False):
    image = Image.fromarray(array)
    if palette:
        image = image.convert("P")
        colors = [0] * (256 * 3)
        colors[7 * 3:7 * 3 + 3] = [255, 0, 0]
        image.putpalette(colors)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def nyu_fixture(tmp_path):
    def create(*, subset=False, split="train", predecessor_test=False):
        source = tmp_path / "source.h5"
        archive = tmp_path / "annotations.zip"
        metadata = tmp_path / "metadata.csv"
        split_path = tmp_path / "split.json"
        # Annotation 1 belongs to h5 index 1. Index 0 is a different scene,
        # image and filled depth, so copying the old N-1 path cannot pass.
        scenes = ["scene_previous", "scene_correct", "scene_distractor"]
        x, y = np.mgrid[:640, :480]
        rgb = np.stack([x % 251, y % 251, np.full_like(x, 81)]).astype(np.uint8)
        raw = np.full((640, 480), 1.25, dtype=np.float32)
        raw[20, 15] = 0.0
        raw[21, 15] = np.nan
        raw[22, 15] = 10.01
        depths = np.full((640, 480), 2.0, dtype=np.float32)
        previous = np.full((640, 480), 3.0, dtype=np.float32)
        labels = np.ones((640, 480), dtype=np.uint16)
        rows = [dict(annotation_id="1", scene="scene_correct", split=split)]
        if predecessor_test:
            # A different annotation in the same predecessor scene is frozen
            # to test, so reading depths_previous for train regression is forbidden.
            scenes[2] = scenes[0]
            rows.append(dict(annotation_id="2", scene="scene_previous", split="test"))
        split_path.write_text(json.dumps({"mapping": rows}), encoding="utf-8")
        fields = ["sample_id", "instance_id", "has_mirror_plane", "has_precise_refined_depth",
                  "has_precise_mask", "scene_id"]
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(sample_id=row["annotation_id"], instance_id="7",
                                     has_mirror_plane="1", has_precise_refined_depth="1",
                                     has_precise_mask="1", scene_id="historical_wrong_scene"))
        with h5py.File(source, "w") as handle:
            string_type = h5py.string_dtype("utf-8")
            handle.create_dataset("scenes", data=scenes, dtype=string_type)
            handle.create_dataset("names", data=["wall", "mirror", "glass"], dtype=string_type)
            if subset:
                handle.attrs["schema"] = "trustprompt.nyu_subset.v1"
                handle.attrs["complete"] = 1
                frame = handle.create_group("frames").create_group("1")
                frame.attrs.update(source_index=1, scene="scene_correct", split=split)
                for key, value in (("images", rgb), ("rawDepths", raw), ("depths", depths),
                                   ("labels", labels), ("depths_previous", previous)):
                    frame.create_dataset(key, data=value, compression="gzip")
            else:
                handle.create_dataset("images", data=np.stack([np.full_like(rgb, 29), rgb, np.full_like(rgb, 191)]),
                                      compression="gzip")
                handle.create_dataset("rawDepths", data=np.stack([np.full_like(raw, 7), raw, np.full_like(raw, 8)]),
                                      compression="gzip")
                handle.create_dataset("depths", data=np.stack([previous, depths, np.full_like(depths, 4)]),
                                      compression="gzip")
                handle.create_dataset("labels", data=np.stack([labels] * 3), compression="gzip")
        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[30:34, 40:45] = 7
        delta = np.zeros(SHAPE, dtype=np.uint16)
        delta[mask == 7] = 65536 - 500  # -500 mm encoded modulo uint16.
        plane = [{"mask_id": 7, "plane": [0, 0, 1, -1500]}]
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("nyu/delta_depth_precise/1.png", _png(delta))
            z.writestr("nyu/mirror_instance_mask_precise/1.png", _png(mask, palette=True))
            z.writestr("nyu/mirror_plane/1.json", json.dumps(plane))
        return {"source": source, "annotation_zip": archive, "metadata_csv": metadata,
                "split_file": split_path}, {"raw": raw, "rgb": rgb, "depths": depths,
                                            "mask": mask, "delta": delta}
    return create


@pytest.mark.parametrize("subset", [False, True])
def test_annotation_n_uses_correct_source_frame_crop_raw_prompt_and_plane(nyu_fixture, subset):
    paths, expected = nyu_fixture(subset=subset)
    sample, audit = Mirror3DNYU(**paths).load("1", regression=True)
    assert sample.frame == "1" and sample.scene == "scene_correct"
    assert sample.metadata["source_index"] == 1
    assert sample.metadata["index_rule"] == "annotation_N_equals_h5_index_N"
    np.testing.assert_array_equal(sample.rgb, expected["rgb"].transpose(2, 1, 0)[CROP])
    np.testing.assert_array_equal(sample.prompt, expected["raw"].T[CROP])
    assert sample.rgb.shape == (456, 608, 3)
    assert tuple(sample.rgb[0, 0]) == (16, 12, 81)
    assert sample.prompt[0, 0] == 1.25
    assert not sample.prompt_valid[3, 4]  # Source x=20, y=15 is a raw hole.
    assert sample.prompt[3, 4] == 0 and sample.gt[3, 4] == 2
    assert not sample.prompt_valid[3, 5] and np.isnan(sample.prompt[3, 5])
    assert not sample.prompt_valid[3, 6] and sample.prompt[3, 6] > 10
    assert sample.metadata["prompt_overrange_count"] == 1
    np.testing.assert_allclose(sample.K_rgb, REGISTERED_K)
    np.testing.assert_allclose(sample.K_prompt, REGISTERED_K)
    assert audit["N"]["count"] == 20
    assert audit["N"]["within_1mm_fraction"] == 1.0
    assert audit["N_minus_1"]["within_1mm_fraction"] == 0.0
    assert audit["N_minus_1"]["median_error_mm"] == pytest.approx(1000, abs=0.001)


def test_uint16_negative_delta_wrap_preserves_official_floor_and_units():
    source = np.array([[2.0, 1.0009, 65.535]], dtype=np.float32)
    delta = np.array([[65536 - 500, 65535, 1]], dtype=np.uint16)
    # Floor 1000.9 -> 1000 before adding -1. Final 65535+1 wraps to zero.
    actual = reconstruct_refined(source, delta)
    np.testing.assert_allclose(actual, [[1.5, 0.999, 0.0]], atol=2e-7, rtol=0)
    assert actual.dtype == np.float32


def test_palette_instance_id_is_preserved_and_gt_background_declared(nyu_fixture):
    paths, expected = nyu_fixture()
    with zipfile.ZipFile(paths["annotation_zip"]) as archive:
        mask = read_png(archive, "nyu/mirror_instance_mask_precise/1.png")
    np.testing.assert_array_equal(mask, expected["mask"])
    assert set(np.unique(mask)) == {0, 7}
    sample, _ = Mirror3DNYU(**paths).load("1")
    assert np.all(sample.material[mask == 7] == 2)
    # An object-semantic "wall" label does not certify ordinary material.
    assert np.all(sample.material[mask == 0] == 0)
    np.testing.assert_allclose(sample.gt[mask == 7], 1.5, atol=2e-7)
    np.testing.assert_allclose(sample.gt[mask == 0], 2.0)
    assert sample.metadata["gt_scope"] == "mirror_refined_background_inpainted"
    assert sample.metadata["background_is_independent_gt"] is False
    assert "rawDepths" in sample.metadata["prompt_source"]
    assert sample.metadata["depth_unit"] == "m"
    assert sample.metadata["depth_convention"] == "axial_z"
    assert sample.metadata["strict_rgb_z_qualified"] is False


def test_corrected_scene_must_match_frozen_split(nyu_fixture):
    paths, _ = nyu_fixture()
    path = paths["split_file"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["mapping"][0]["scene"] = "scene_previous"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="corrected source scene"):
        Mirror3DNYU(**paths).load("1")


def test_test_split_rejected_before_opening_hdf5(nyu_fixture, monkeypatch):
    paths, _ = nyu_fixture(split="test")
    loader = Mirror3DNYU(**paths)
    def forbidden_open(*args, **kwargs):
        raise AssertionError("test payload container must not be opened")
    monkeypatch.setattr(h5py, "File", forbidden_open)
    with pytest.raises(ValueError, match="test payload access disabled"):
        loader.load("1")


def test_missing_raw_depths_fails_instead_of_using_filled_depths(nyu_fixture):
    paths, _ = nyu_fixture()
    with h5py.File(paths["source"], "r+") as handle:
        del handle["rawDepths"]
    with pytest.raises(ValueError, match="rawDepths required"):
        Mirror3DNYU(**paths).load("1")


@pytest.mark.parametrize("subset", [False, True])
def test_regression_cannot_read_predecessor_from_test_scene(nyu_fixture, subset):
    paths, _ = nyu_fixture(subset=subset, predecessor_test=True)
    loader = Mirror3DNYU(**paths)
    # Normal train access still works; only predecessor regression is rejected.
    assert loader.load("1")[0].split == "train"
    with pytest.raises(ValueError, match="predecessor.*test scene"):
        loader.load("1", regression=True)


@pytest.mark.parametrize("field,value,error", [
    ("complete", 0, "unqualified/incomplete"),
    ("source_index", 0, "subset frame identity mismatch"),
    ("split", "val", "subset frame identity mismatch"),
])
def test_subset_cache_identity_and_completion_guards(nyu_fixture, field, value, error):
    paths, _ = nyu_fixture(subset=True)
    with h5py.File(paths["source"], "r+") as handle:
        target = handle if field == "complete" else handle["frames"]["1"]
        target.attrs[field] = value
    with pytest.raises(ValueError, match=error):
        Mirror3DNYU(**paths).load("1")


@pytest.mark.parametrize("bad_id", ["0", "01", "1449", "-1", "1.0"])
def test_noncanonical_or_out_of_range_annotation_id_rejected(bad_id):
    with pytest.raises(ValueError):
        annotation_index(bad_id)
