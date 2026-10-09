"""Image -> scene -> dataset aggregation without hidden invalid-value drops."""
from __future__ import annotations

from collections import defaultdict
import math

from .metrics import METRIC_NAMES


def _strict_mean(values):
    """Only explicit empty domains may be excluded by the caller."""
    if not values or any(value is None for value in values):
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite JSON metric; encode invalid results as None with status")
    return float(sum(values) / len(values))


def _combine_domain(domains, names):
    active = [domain for domain in domains if domain["count"] > 0]
    count = sum(domain["count"] for domain in domains)
    invalid_count = sum(domain["invalid_count"] for domain in domains)
    failures = [domain["status"] for domain in active if domain["status"] != "ok"]
    status = ("empty" if not active else "invalid_predictions" if invalid_count else
              failures[0] if failures else "ok")
    result = {
        "count": int(count), "invalid_count": int(invalid_count),
        # Pooled rate is diagnostic only. Accuracy metrics use hierarchical means.
        "invalid_rate": invalid_count / count if count else None,
        "status": status,
        "unit_count": len(domains), "nonempty_unit_count": len(active),
        "empty_unit_count": len(domains) - len(active),
        "invalid_unit_count": len(failures),
    }
    result.update({name: _strict_mean([domain[name] for domain in active]) for name in names})
    return result


def _combine(metrics):
    if not metrics:
        raise ValueError("cannot aggregate zero metric records")
    protocol = metrics[0]["protocol"]
    region_names = set(metrics[0]["regions"])
    if any(m["protocol"] != protocol or set(m["regions"]) != region_names for m in metrics):
        raise ValueError("all records must use the same protocol and region definitions")
    return {
        "protocol": protocol,
        "regions": {
            name: _combine_domain([m["regions"][name] for m in metrics], METRIC_NAMES)
            for name in sorted(region_names)
        },
        **{name: _combine_domain([m[name] for m in metrics], ("value", "lower_bound", "upper_bound"))
           for name in ("IR", "DOR")},
    }


def aggregate(records):
    """Aggregate records with dataset, scene, frame, split, metrics keys.

    Return ``{'datasets': {name: {'scenes': {scene: {'metrics': ...}},
    'metrics': ..., 'image_count': ..., 'scene_count': ...}}}``. Each dataset's
    metrics first average image metrics equally within each nonempty scene,
    then average scenes equally. Empty domains are explicit NA; any nonempty
    invalid metric propagates None through both levels. Pixel counts are sums
    for auditing, not weights. This function rejects mixed splits/datasets in
    one scene, duplicate frame/sensor records, and mixed prediction methods.
    """
    records = list(records)
    if not records:
        raise ValueError("no per-image records to aggregate")
    methods = {record.get("method") for record in records}
    if len(methods) > 1:
        raise ValueError("aggregate one method at a time")
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    splits = {}
    for record in records:
        for key in ("dataset", "scene", "frame", "split", "metrics"):
            if key not in record:
                raise ValueError(f"record is missing {key}")
        key = (record["dataset"], record["scene"], record["frame"], record.get("sensor"))
        if key in seen:
            raise ValueError(f"duplicate frame/sensor record: {key}")
        seen.add(key)
        scene_key = (record["dataset"], record["scene"])
        if scene_key in splits and splits[scene_key] != record["split"]:
            raise ValueError(f"scene crosses train/validation/test splits: {scene_key}")
        splits[scene_key] = record["split"]
        groups[record["dataset"]][record["scene"]].append(record)
    output = {"aggregation": "equal_images_within_scene_then_equal_nonempty_scenes",
              "invalid_policy": "propagate_nonempty_invalid_metrics",
              "datasets": {}}
    for dataset, scenes in sorted(groups.items()):
        # Mixing different splits in a dataset summary would make development
        # and test results indistinguishable; make callers evaluate separately.
        dataset_splits = {record["split"] for rows in scenes.values() for record in rows}
        if len(dataset_splits) != 1:
            raise ValueError("aggregate one dataset split at a time")
        scene_results = {
            scene: {"image_count": len(rows), "split": rows[0]["split"],
                    "metrics": _combine([row["metrics"] for row in rows])}
            for scene, rows in sorted(scenes.items())
        }
        output["datasets"][dataset] = {
            "image_count": sum(len(rows) for rows in scenes.values()),
            "scene_count": len(scenes), "split": next(iter(dataset_splits)),
            "scenes": scene_results,
            "metrics": _combine([result["metrics"] for result in scene_results.values()]),
        }
    return output
