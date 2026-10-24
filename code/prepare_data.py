"""O1: build comparable PlantVillage (lab) and PlantDoc (field) sets on shared classes.

Outputs (in data_out/):
  class_map.csv            PlantDoc label -> PlantVillage folder -> unified class id
  pv_split.csv             PlantVillage pilot train/val/test manifest (stratified, seed 42)
  pd_crops/<split>/...     cropped PlantDoc leaves from bounding boxes
  pd_manifest.csv          PlantDoc crop manifest (train pool / test)
  data_summary.csv         per-class counts
"""
import argparse, os, random, csv
import pandas as pd
from PIL import Image

# PlantDoc label -> PlantVillage folder (27 classes present in the PlantDoc test set)
CLASS_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}
MIN_CROP = 32  # px; smaller boxes are excluded (too little leaf to classify)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    classes = sorted(CLASS_MAP.values())
    cid = {c: i for i, c in enumerate(classes)}
    with open(f"{a.out}/class_map.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class_id", "plantvillage", "plantdoc"])
        for pdl, pvl in sorted(CLASS_MAP.items(), key=lambda x: cid[x[1]]):
            w.writerow([cid[pvl], pvl, pdl])

    # ---- PlantVillage: stratified per-class split (pilot caps) ----
    rows = []
    for c in classes:
        files = sorted(os.listdir(os.path.join(a.pv, c)))
        files = [x for x in files if x.lower().endswith((".jpg", ".jpeg", ".png"))]
        n_total = len(files)
        rng.shuffle(files)
        n_tr, n_va, n_te = a.pv_train, a.pv_val, a.pv_test
        parts = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va],
                 "test": files[n_tr + n_va:n_tr + n_va + n_te]}
        for s, fl in parts.items():
            for x in fl:
                rows.append({"path": os.path.join(a.pv, c, x), "class": c,
                             "label": cid[c], "split": s, "pv_class_total": n_total})
    pv = pd.DataFrame(rows); pv.to_csv(f"{a.out}/pv_split.csv", index=False)

    # ---- PlantDoc: crop bounding boxes; TRAIN -> fine-tuning pool, TEST -> field test ----
    crows, skipped = [], {"unmapped": 0, "too_small": 0, "missing_file": 0}
    for split, csvname, folder in [("pool", "train_labels.csv", "TRAIN"),
                                   ("test", "test_labels.csv", "TEST")]:
        df = pd.read_csv(os.path.join(a.pd, csvname))
        for i, r in df.iterrows():
            if r["class"] not in CLASS_MAP: skipped["unmapped"] += 1; continue
            p = os.path.join(a.pd, folder, r["filename"])
            if not os.path.exists(p): skipped["missing_file"] += 1; continue
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                skipped["missing_file"] += 1; continue
            W, H = im.size
            sx, sy = 1.0, 1.0
            if int(r.width) <= 0 or int(r.height) <= 0:
                pass  # size not recorded -> keep raw coordinates, clip below
            elif (int(r.width), int(r.height)) != (W, H):
                # annotation was made on a resized copy of the image -> rescale the box
                sx, sy = W / float(r.width), H / float(r.height)
                skipped["rescaled_box"] = skipped.get("rescaled_box", 0) + 1
            x0, y0 = max(0, int(r.xmin * sx)), max(0, int(r.ymin * sy))
            x1, y1 = min(W, int(r.xmax * sx)), min(H, int(r.ymax * sy))
            if x1 <= x0 or y1 <= y0: skipped["invalid_box"] = skipped.get("invalid_box", 0) + 1; continue
            if min(x1 - x0, y1 - y0) < MIN_CROP: skipped["too_small"] += 1; continue
            crop = im.crop((x0, y0, x1, y1))
            c = CLASS_MAP[r["class"]]
            d = os.path.join(a.out, "pd_crops", split, str(cid[c])); os.makedirs(d, exist_ok=True)
            outp = os.path.join(d, f"{os.path.splitext(r['filename'])[0][:60]}_{i}.jpg")
            crop.save(outp, quality=95)
            crows.append({"path": outp, "class": c, "label": cid[c], "split": split,
                          "source_image": r["filename"], "w": x1 - x0, "h": y1 - y0})
    pdm = pd.DataFrame(crows); pdm.to_csv(f"{a.out}/pd_manifest.csv", index=False)

    summ = pd.DataFrame({"class": classes})
    summ["class_id"] = summ["class"].map(cid)
    summ["pv_available"] = summ["class"].map(pv.groupby("class").pv_class_total.first())
    for s in ["train", "val", "test"]:
        summ[f"pv_{s}"] = summ["class"].map(pv[pv.split == s].groupby("class").size()).fillna(0).astype(int)
    for s in ["pool", "test"]:
        summ[f"pd_{s}"] = summ["class"].map(pdm[pdm.split == s].groupby("class").size()).fillna(0).astype(int)
    summ.to_csv(f"{a.out}/data_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("skipped boxes:", skipped)
    print("totals:", summ[[c for c in summ.columns if c.startswith(("pv_", "pd_"))]].sum().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv", default="../data/PlantVillage-Dataset/raw/color")
    ap.add_argument("--pd", default="../data/PlantDoc-Object-Detection-Dataset")
    ap.add_argument("--out", default="data_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pv_train", type=int, default=100)
    ap.add_argument("--pv_val", type=int, default=20)
    ap.add_argument("--pv_test", type=int, default=40)
    main(ap.parse_args())
