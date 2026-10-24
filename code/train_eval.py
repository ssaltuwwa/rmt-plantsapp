"""O2/O3 pilot: train MobileNetV3-Large on PlantVillage (lab), evaluate on PlantVillage test and
PlantDoc field test; optional field-style augmentation and few-shot fine-tuning on PlantDoc crops.

Conditions:
  baseline : standard augmentation, PlantVillage train only
  fieldaug : field-style augmentation, PlantVillage train only
  ftK      : start from baseline checkpoint, fine-tune on K PlantDoc crops/class (leak-filtered pool)
Model selection: best epoch by PlantVillage *validation* macro-F1; fine-tuning uses a fixed schedule.
The PlantDoc test set is never used for training, model selection or early stopping.
"""
import argparse, json, os, random, sys, time, platform
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

MBV3 = os.environ.get("MBV3_DIR", "/home/claude/data/mobilenetv3.pytorch")  # clone of github.com/d-li14/mobilenetv3.pytorch
sys.path.insert(0, MBV3)
from mobilenetv3 import mobilenetv3_large  # ImageNet weights: pretrained/mobilenetv3-large-1cd25616.pth

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NCLS = 27


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class DS(Dataset):
    def __init__(self, df, tf): self.p, self.y, self.tf = df.path.tolist(), df.label.tolist(), tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]


def tfs(kind, size):
    if kind == "standard":
        return T.Compose([T.RandomResizedCrop(size, scale=(0.8, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    if kind == "field":  # simulates field capture: framing, rotation, lighting, blur, noise-like jitter
        return T.Compose([T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.6, 1.6)),
                          T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30),
                          T.ColorJitter(0.5, 0.5, 0.4, 0.05),
                          T.RandomApply([T.GaussianBlur(7, sigma=(0.5, 3.0))], p=0.5),
                          T.RandomGrayscale(p=0.05), T.ToTensor(), T.Normalize(MEAN, STD),
                          T.RandomErasing(p=0.3, scale=(0.02, 0.15))])
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model():
    m = mobilenetv3_large()
    m.load_state_dict(torch.load(f"{MBV3}/pretrained/mobilenetv3-large-1cd25616.pth",
                                 map_location="cpu"))
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NCLS)
    return m


@torch.no_grad()
def predict(m, df, size, bs=64):
    m.eval(); dl = DataLoader(DS(df, tfs("eval", size)), batch_size=bs, num_workers=2)
    P = []
    for x, _ in dl: P.append(torch.softmax(m(x), 1))
    return torch.cat(P).numpy()


def metrics(y, prob, groups=None, B=1000, seed=0):
    pred = prob.argmax(1)
    out = {"n": int(len(y)), "accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=list(range(NCLS)), zero_division=0),
           "top3_accuracy": float(np.mean([yy in np.argsort(-p)[:3] for yy, p in zip(y, prob)]))}
    # cluster bootstrap over source images (crops of one photo are not independent)
    rng = np.random.default_rng(seed)
    g = np.asarray(groups if groups is not None else np.arange(len(y)))
    ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    bs = []
    for _ in range(B):
        s = np.concatenate([idx[k] for k in rng.choice(ug, len(ug), replace=True)])
        bs.append(f1_score(y[s], pred[s], average="macro", labels=list(range(NCLS)), zero_division=0))
    out["macro_f1_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def run_epochs(m, dl, opt, sched, epochs, val_df, size, log, tag):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train(); t0 = time.time(); tl, n = 0.0, 0
        for x, yb in dl:
            opt.zero_grad(); loss = crit(m(x), yb); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(yb); n += len(yb)
        rec = {"stage": tag, "epoch": ep + 1, "train_loss": tl / n, "sec": round(time.time() - t0, 1)}
        if val_df is not None:
            pv = predict(m, val_df, size)
            rec["val_macro_f1"] = f1_score(val_df.label, pv.argmax(1), average="macro")
            if rec["val_macro_f1"] > best:
                best = rec["val_macro_f1"]; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        log.append(rec); print(json.dumps(rec), flush=True)
    if best_state is not None: m.load_state_dict(best_state)
    return m


def main(a):
    set_seed(a.seed); torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    pv = pd.read_csv(f"{a.data}/pv_split.csv"); pdm = pd.read_csv(f"{a.data}/pd_manifest.csv")
    pd_test = pdm[pdm.split == "test"].reset_index(drop=True)
    pv_tr, pv_va, pv_te = (pv[pv.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    log = []
    cfg = vars(a) | {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "python": platform.python_version(), "cpu_threads": a.threads}

    if a.condition in ("baseline", "fieldaug"):
        m = build_model()
        dl = DataLoader(DS(pv_tr, tfs("standard" if a.condition == "baseline" else "field", a.size)),
                        batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
        m = run_epochs(m, dl, opt, sched, a.epochs, pv_va, a.size, log, "pv_train")
    else:  # ftK
        k = int(a.condition[2:])
        m = build_model(); m.load_state_dict(torch.load(a.init, map_location="cpu"))
        excl = set(pd.read_csv(f"{a.data}/leakage_duplicates_excluded_pool_images.csv").pool_source_image)
        pool = pdm[(pdm.split == "pool") & (~pdm.source_image.isin(excl))]
        ft = pd.concat([g.sample(min(k, len(g)), random_state=a.seed)
                        for _, g in pool.groupby("label")]).reset_index(drop=True)
        cfg["ft_images"] = int(len(ft)); cfg["ft_excluded_pool_images"] = len(excl)
        dl = DataLoader(DS(ft, tfs("standard", a.size)), batch_size=min(a.bs, len(ft)),
                        shuffle=True, num_workers=2, drop_last=False)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
        m = run_epochs(m, dl, opt, sched, a.epochs, None, a.size, log, f"ft{k}")
        ft.to_csv(f"{a.out}/ft_images_used.csv", index=False)

    torch.save(m.state_dict(), f"{a.out}/model.pt")
    res = {"config": cfg, "train_log": log}
    for name, df, grp in [("plantvillage_test", pv_te, None), ("plantdoc_test", pd_test, pd_test.source_image.values)]:
        prob = predict(m, df, a.size); y = df.label.values
        res[name] = metrics(y, prob, grp)
        p, r, f, s = precision_recall_fscore_support(y, prob.argmax(1), labels=list(range(NCLS)), zero_division=0)
        pd.DataFrame({"label": range(NCLS), "precision": p, "recall": r, "f1": f, "support": s}).to_csv(
            f"{a.out}/{name}_per_class.csv", index=False)
        np.savetxt(f"{a.out}/{name}_confusion.csv", confusion_matrix(y, prob.argmax(1), labels=list(range(NCLS))),
                   fmt="%d", delimiter=",")
        out = df[["path", "label"]].copy(); out["pred"] = prob.argmax(1); out["conf"] = prob.max(1)
        out.to_csv(f"{a.out}/{name}_predictions.csv", index=False)
        print(name, json.dumps(res[name]), flush=True)

    # CPU latency, batch size 1 (relevant to O4: < 2 s per photo)
    m.eval(); x = torch.randn(1, 3, a.size, a.size); ts = []
    with torch.no_grad():
        for i in range(60):
            t0 = time.perf_counter(); m(x); ts.append(time.perf_counter() - t0)
    res["cpu_latency_ms"] = {"median": 1000 * float(np.median(ts[10:])), "p95": 1000 * float(np.percentile(ts[10:], 95))}
    res["params_millions"] = sum(p.numel() for p in m.parameters()) / 1e6
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print("latency", res["cpu_latency_ms"], "params(M)", round(res["params_millions"], 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--data", default="../data_out"); ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--threads", type=int, default=2)
    main(ap.parse_args())
