"""Mirror3D NYU adapter, annotation N == h5 index N (NOT N-1).

Selective adaptation of GLARE's prepare_rgb_first_surface_clear_mirror.py:
crop, indexed mask reading and uint16 modular delta reconstruction only.
The old adapter's N-1 indexing, scene IDs and processed caches are not reused.
"""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import zipfile
import h5py
import numpy as np
from PIL import Image
from .schema import Sample, validate_scene_splits

CROP = (slice(12,468), slice(16,624))
SHAPE = (456,608)
PLANE_K = np.array([[519.,0,304.],[0,519.,228.],[0,0,1.]])
# NYU RGB calibration mapped to the released registered-depth array indices.
# See docs/intrinsics_audit.md: toolbox rectification/projection differ by 1px.
REGISTERED_K = np.array([[518.8579011745019,0,325.58244941119034-1-16],
                         [0,519.4696111212749,253.73616633400465-1-12],[0,0,1.]])


def annotation_index(annotation_id: str | int) -> int:
    n=int(annotation_id)
    if str(n) != str(annotation_id) or not 0 < n < 1449:
        raise ValueError("annotation ID must be canonical, with source index N in 1..1448")
    return n


def reconstruct_refined_mm(depths_m, delta_mm):
    source=np.asarray(depths_m)
    if source.shape != delta_mm.shape or np.any(~np.isfinite(source) | (source < 0) | (source*1000 >= 65536)):
        raise ValueError("invalid source for uint16 millimetre reconstruction")
    # Preserve official float32 multiplication and floor-to-uint16 behaviour.
    base=(source.astype(np.float32)*np.float32(1000)).astype(np.uint16)
    return ((base.astype(np.uint32)+delta_mm.astype(np.uint32)) & 65535).astype(np.uint16)


def reconstruct_refined(depths_m, delta_mm):
    return reconstruct_refined_mm(depths_m, delta_mm).astype(np.float32)*np.float32(.001)


def read_png(archive, name):
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return np.array(im)  # Do NOT convert palette IDs to colors.


def plane_regression(depths_m, previous_m, delta, labels, planes):
    y,x=np.mgrid[:SHAPE[0],:SHAPE[1]]
    results={}
    for name, source in (("N",depths_m),("N_minus_1",previous_m)):
        refined=reconstruct_refined_mm(source,delta).astype(np.float64)
        errors=[]
        instance_rows=[]
        for plane in planes:
            a,b,c,d=plane["plane"]
            with np.errstate(divide="ignore",invalid="ignore"):
                z=-d/(a*(x-304)/519+b*(y-228)/519+c)
            mask=labels==int(plane["mask_id"])
            e=np.abs(refined[mask]-z[mask])
            errors.append(e)
            instance_rows.append(dict(instance=int(plane["mask_id"]),count=int(e.size),
                                      within_1mm=int(np.count_nonzero(e<=1.0)),
                                      nonpositive_plane=int(np.count_nonzero(z[mask]<=0))))
        e=np.concatenate(errors)
        results[name]=dict(count=int(e.size),within_1mm=int(np.count_nonzero(e<=1.0)),
                           within_1mm_fraction=float(np.count_nonzero(e<=1.0)/e.size),
                           median_error_mm=float(np.median(e)),instances=instance_rows)
    return results


class Mirror3DNYU:
    """Reads complete official MAT or provenance-checked TrustPrompt subset.

    Test frames require an explicit evaluation flag. The first preparation
    command reads training frames only. Splits are frozen before depth reads.
    """
    def __init__(self, source, annotation_zip, metadata_csv, split_file, *, allow_test=False):
        self.source,self.annotation_zip=Path(source),Path(annotation_zip)
        self.allow_test=allow_test
        split=json.loads(Path(split_file).read_text(encoding="utf-8"))
        self.mapping={str(r["annotation_id"]):r for r in split["mapping"]}
        validate_scene_splits([dict(dataset="mirror3d_nyu",scene=r["scene"],frame=str(r["annotation_id"]),split=r["split"])
                               for r in split["mapping"]])
        self.instances={}
        with Path(metadata_csv).open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                if any(r[k]!="1" for k in ("has_mirror_plane","has_precise_refined_depth","has_precise_mask")):
                    raise ValueError("unqualified annotation in metadata")
                self.instances.setdefault(r["sample_id"],set()).add(int(r["instance_id"]))
        if set(self.mapping)!=set(self.instances):
            raise ValueError("split/CSV sample IDs differ")

    @staticmethod
    def _strings(handle,key):
        d=handle[key]
        if h5py.check_dtype(ref=d.dtype) is not None:
            return [''.join(chr(int(c)) for c in handle[r][:].ravel()) for r in d[:].ravel()]
        return list(d.asstr()[:].ravel())

    def load(self, annotation_id, *, regression=False):
        sid=str(annotation_id); n=annotation_index(sid)
        record=self.mapping[sid]
        if record["split"]=="test" and not self.allow_test:
            raise ValueError("test payload access disabled in development")
        with h5py.File(self.source,"r") as h:
            subset="frames" in h
            if subset and (h.attrs.get("schema")!="trustprompt.nyu_subset.v1" or h.attrs.get("complete")!=1):
                raise ValueError("unqualified/incomplete TrustPrompt source cache")
            if not subset and "rawDepths" not in h:
                raise ValueError("rawDepths required; legacy selected cache cannot supply prompts")
            scenes=self._strings(h,"scenes")
            names=self._strings(h,"names")
            if scenes[n]!=record["scene"]:
                raise ValueError("corrected source scene disagrees with frozen split")
            if regression and not self.allow_test and any(r["scene"]==scenes[n-1] and r["split"]=="test" for r in self.mapping.values()):
                raise ValueError("regression predecessor belongs to a test scene")
            if subset:
                g=h["frames"][sid]
                if int(g.attrs["source_index"])!=n or g.attrs["scene"]!=scenes[n] or g.attrs["split"]!=record["split"]:
                    raise ValueError("subset frame identity mismatch")
                arrays={key:g[key][:] for key in ("images","rawDepths","depths","labels")}
                previous=g["depths_previous"][:] if regression else None
            else:
                arrays={key:h[key][n] for key in ("images","rawDepths","depths","labels")}
                previous=h["depths"][n-1] if regression else None
        if arrays["images"].shape!=(3,640,480) or any(arrays[k].shape!=(640,480) for k in ("rawDepths","depths","labels")):
            raise ValueError("NYU native layout mismatch")
        rgb=np.ascontiguousarray(arrays["images"].transpose(2,1,0)[CROP])
        prompt=np.ascontiguousarray(arrays["rawDepths"].T[CROP],dtype=np.float32)
        source=arrays["depths"].T[CROP]
        semantic=arrays["labels"].T[CROP]
        with zipfile.ZipFile(self.annotation_zip) as z:
            delta=read_png(z,f"nyu/delta_depth_precise/{sid}.png")
            mask=read_png(z,f"nyu/mirror_instance_mask_precise/{sid}.png")
            planes=json.loads(z.read(f"nyu/mirror_plane/{sid}.json"))
        if mask.shape!=SHAPE or delta.shape!=SHAPE:
            raise ValueError("Mirror3D crop mismatch")
        ids=set(np.unique(mask).tolist())-{0}
        if ids!=self.instances[sid] or ids!={int(p["mask_id"]) for p in planes}:
            raise ValueError("mask/CSV/plane instance IDs differ")
        gt=reconstruct_refined(source,delta)
        # Object semantics do not establish physical material. Until the mapping
        # is audited, only precise mirror instances receive a material class.
        ordinary_ids=[]
        material=np.zeros(SHAPE,np.uint8)
        material[np.isin(semantic,ordinary_ids)]=1
        material[mask>0]=2
        pv=np.isfinite(prompt)&(prompt>0)&(prompt<=10.)
        gv=np.isfinite(gt)&(gt>0)
        sample=Sample(rgb,prompt,pv,gt,gv,material,REGISTERED_K.copy(),REGISTERED_K.copy(),
                      "kinect1","mirror3d_nyu",scenes[n],sid,record["split"],metadata=dict(
            depth_unit="m",depth_convention="axial_z",synthetic=False,source_index=n,
            index_rule="annotation_N_equals_h5_index_N",crop_yxhw=[12,16,456,608],
            prompt_source="NYU rawDepths; RGB-registered unfilled Kinect sensor axial Z",
            prompt_max_m=10.,prompt_overrange_count=int(np.count_nonzero(np.isfinite(prompt)&(prompt>10))),
            gt_source="uint16(depths*1000)+delta_depth_precise modulo 65536",
            gt_scope="mirror_refined_background_inpainted",background_is_independent_gt=False,
            camera_reference="NYU released registered-Z; exact RGB-axis Z not certified",
            strict_rgb_z_qualified=False,
            K_convention="NYU calibrated projection indexing minus 1 then crop; rectification 1px caveat",
            annotation_plane_K=PLANE_K.tolist(),annotation_plane_units="mm",
            material_source="precise mirror mask only; all other material unknown pending taxonomy audit",
            ordinary_semantic_ids=ordinary_ids,delta_nonzero_outside_mirror=int(np.count_nonzero((mask==0)&(delta!=0)))))
        audit=plane_regression(source,previous.T[CROP],delta,mask,planes) if regression else None
        return sample.validate(),audit
