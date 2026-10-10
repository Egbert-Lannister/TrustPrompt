"""Resume planning for interrupted NYU Range acquisitions (no network)."""
from scripts.fetch_nyu_subset import plan_resume

FIELDS = ["images", "depths", "rawDepths", "labels", "depths_previous"]


def payloads(frame, digests):
    return [{"annotation_id": int(frame), "field": f, "sha256": d} for f, d in digests.items()]


def test_complete_matching_frames_are_kept_partial_and_mismatched_are_rewritten():
    full = {f: f"h_{f}" for f in FIELDS}
    existing = {"43": dict(full), "347": {"images": "h_images", "depths": "h_depths"},
                "48": {**full, "rawDepths": "corrupted"}}
    prior = payloads("43", full) + payloads("347", {"images": "h_images", "depths": "h_depths"}) + payloads("48", full)
    keep, rewrite = plan_resume(existing, prior, lambda frame: FIELDS)
    assert keep == ["43"]
    assert rewrite == ["48", "347"]


def test_frames_without_predecessor_need_only_four_fields():
    four = {f: f"h_{f}" for f in FIELDS[:4]}
    keep, rewrite = plan_resume({"667": four}, payloads("667", four),
                                lambda frame: FIELDS[:4] if frame == "667" else FIELDS)
    assert keep == ["667"] and rewrite == []


def test_unrecorded_frame_is_never_trusted():
    full = {f: "x" for f in FIELDS}
    keep, rewrite = plan_resume({"99": full}, [], lambda frame: FIELDS)
    assert keep == [] and rewrite == ["99"]
