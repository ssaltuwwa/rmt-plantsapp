"""Leakage check: are any PlantDoc test leaves duplicated in the fine-tuning pool?
(1) identical source-image file names across TRAIN/TEST,
(2) near-duplicate crops by perceptual hash (pHash Hamming distance <= THR)."""
import sys, pandas as pd, imagehash
from PIL import Image

THR = 4
m = pd.read_csv(sys.argv[1])
pool, test = m[m.split == "pool"].copy(), m[m.split == "test"].copy()
same_src = set(pool.source_image) & set(test.source_image)
h = lambda p: imagehash.phash(Image.open(p).convert("RGB"))
pool["h"] = pool.path.map(h); test["h"] = test.path.map(h)
ph = list(zip(pool.path, pool.h))
dups = []
for tp, th in zip(test.path, test.h):
    best = min(((th - hh), pp) for pp, hh in ph)
    if best[0] <= THR:
        dups.append({"test_path": tp, "pool_path": best[1], "hamming": best[0]})
d = pd.DataFrame(dups, columns=["test_path", "pool_path", "hamming"])
d.to_csv(sys.argv[2], index=False)
print(f"test crops: {len(test)}, pool crops: {len(pool)}")
print(f"source images present in both TRAIN and TEST: {len(same_src)}")
print(f"test crops with a near-duplicate in pool (pHash dist <= {THR}): {len(d)} "
      f"({100*len(d)/len(test):.1f}%) from {d.test_path.map(lambda p: p.split('/')[-1].rsplit('_',1)[0]).nunique() if len(d) else 0} test images")

# (3) whole-image near-duplicates TRAIN vs TEST -> exclude these source images from the fine-tuning pool
R = sys.argv[3]
th = {f: imagehash.phash(Image.open(f"{R}/TEST/{f}").convert("RGB")) for f in test.source_image.unique()}
excl = []
for f in pool.source_image.unique():
    hf = imagehash.phash(Image.open(f"{R}/TRAIN/{f}").convert("RGB"))
    hit = min((hf - v, k) for k, v in th.items())
    if hit[0] <= THR: excl.append({"pool_source_image": f, "test_source_image": hit[1], "hamming": hit[0]})
crop_dup_src = set(pool[pool.path.isin(d.pool_path)].source_image)
ex = pd.DataFrame(excl, columns=["pool_source_image", "test_source_image", "hamming"])
allx = sorted(set(ex.pool_source_image) | crop_dup_src)
pd.DataFrame({"pool_source_image": allx}).to_csv(sys.argv[2].replace(".csv", "_excluded_pool_images.csv"), index=False)
print(f"TRAIN images that are near-duplicates of a TEST image (whole-image pHash <= {THR}): {len(ex)}")
print(f"pool source images excluded from fine-tuning (union with crop-level matches): {len(allx)}; "
      f"crops removed from pool: {pool.source_image.isin(allx).sum()}")
