"""Fixed-scale alignment panels; display clips do not affect any metric."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def alignment_panel(sample, prompt, pred, masks, path):
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    a=axes.ravel()
    hammer=sample.dataset=="hammer"
    a[0].imshow(sample.rgb);a[0].set_title(f"RGB ({sample.sensor} prompt)" if hammer else "RGB (released crop)")
    a[1].imshow(sample.rgb)
    if hammer and sample.behind is not None:
        focus=sample.gt_valid&(np.abs(sample.gt-sample.behind)>1e-3);label="RGB + object contour (GT vs naked)"
    else:
        focus=sample.material==2;label="RGB + precise mirror contour"
    if focus.any():a[1].contour(focus,levels=[.5],colors=["magenta"],linewidths=.7)
    a[1].set_title(label)
    cm=plt.get_cmap("turbo").copy();cm.set_bad("black")
    vmax=float(np.ceil(np.nanpercentile(np.where(sample.gt_valid,sample.gt,np.nan),99.5)*1.5)) if hammer else 10
    im=a[2].imshow(np.where(np.isfinite(prompt)&(prompt>0),prompt,np.nan),vmin=0,vmax=vmax,cmap=cm)
    a[2].set_title("Raw prompt / holes black [m]")
    a[3].imshow(np.where(sample.gt_valid,sample.gt,np.nan),vmin=0,vmax=vmax,cmap=cm)
    a[3].set_title("Synthetic analytic GT [m]" if sample.metadata["synthetic"] else
                   "Rendered first-surface GT [m]" if hammer else "Refined GT; background inpainted")
    a[4].imshow(pred,vmin=0,vmax=vmax,cmap=cm);a[4].set_title("Prediction [m]")
    labels=np.zeros(sample.gt.shape,dtype=np.uint8)
    labels[masks["clean_all"]]=1;labels[masks["Cfar"]]=2;labels[masks["Cnear"]]=3;labels[masks["hole"]]=4
    a[5].imshow(labels,cmap=ListedColormap(["black","#43a047","#e53935","#f9a825","#1976d2"]),vmin=0,vmax=4)
    a[5].set_title("Green clean / red far / gold near / blue hole")
    a[6].imshow(masks["band"],cmap="gray",vmin=0,vmax=1);a[6].set_title("Fixed pollution boundary band")
    rel=np.full(pred.shape,np.nan)
    valid=sample.gt_valid&np.isfinite(pred)&(pred>0)
    rel[valid]=np.abs(pred[valid]-sample.gt[valid])/sample.gt[valid]
    err=a[7].imshow(rel,cmap="magma",vmin=0,vmax=1);a[7].set_title("AbsRel display clipped at 1 (metrics unclipped)")
    for ax in a:ax.set_axis_off()
    fig.colorbar(im,ax=[a[2],a[3],a[4]],shrink=.6,label="camera-axis depth [m]")
    fig.colorbar(err,ax=a[7],shrink=.6,label="relative error")
    tag="SYNTHETIC ENGINEERING ONLY" if sample.metadata["synthetic"] else f"REAL {sample.split.upper()} / development, not final test"
    fig.suptitle(f"{tag} | {sample.dataset}/{sample.scene}/{sample.frame}",fontsize=14)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=100);plt.close(fig)
