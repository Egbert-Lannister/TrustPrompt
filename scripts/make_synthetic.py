"""Generate explicit synthetic fixtures; these are NEVER real experiments."""
import argparse
from pathlib import Path
import numpy as np
from tp.data.schema import Sample, save_sample
from tp.provenance import write_json, sha256, ROOT


def make_sample(i=0, seed=20261009):
    rng = np.random.default_rng(seed + i)
    h, w = 96, 128
    y, x = np.mgrid[:h, :w]
    gt = (1 / (0.45 + 0.0005 * x - 0.0003 * y)).astype(np.float32)
    rgb = np.stack((x * 255 / w, y * 255 / h, (x + y) * 255 / (w+h)), -1).astype(np.uint8)
    material = np.ones((h,w), np.uint8)
    material[22:70, 35:95] = 2
    prompt = gt[::2,::2].copy()
    prompt *= rng.normal(1, .002, prompt.shape).astype(np.float32)
    prompt[11:35,18:48] *= 1.6
    prompt[37:43,8:16] *= .75
    prompt[5:10,10:22] = 0
    k = np.array([[100,0,63.5],[0,100,47.5],[0,0,1]],dtype=np.float64)
    kp = np.array([[50,0,31.5],[0,50,23.5],[0,0,1]],dtype=np.float64)
    return Sample(rgb,prompt,prompt>0,gt,np.ones((h,w),bool),material,k,kp,
                  "synthetic", "synthetic_fixture", f"scene_{i//2}",str(i),"train",
                  metadata=dict(depth_unit="m",depth_convention="axial_z",synthetic=True,
                                gt_source="analytic tilted plane",gt_scope="synthetic_exact",
                                prompt_grid="half resolution; centre-nearest resize",seed=seed+i)).validate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=ROOT/"data/processed/synthetic_smoke_v1")
    p.add_argument("--seed",type=int,default=20261009)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for i in range(5):
        s=make_sample(i,args.seed)
        path=args.output/f"{i}.npz"
        save_sample(s,path)
        records.append(dict(path=path.name,sha256=sha256(path),dataset=s.dataset,scene=s.scene,frame=s.frame,split=s.split,sensor=s.sensor))
    write_json(args.output/"manifest.json",dict(schema="trustprompt.manifest.v1",synthetic=True,records=records,seed=args.seed))
    print(f"Wrote {len(records)} SYNTHETIC engineering samples to {args.output}")


if __name__=="__main__":
    main()
