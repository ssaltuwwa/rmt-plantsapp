"""Collect pilot runs -> summary table, figures, selective-prediction ('not sure') table."""
import json, os, csv, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

R, F = "../runs", "../figs"; os.makedirs(F, exist_ok=True)
ORDER = [("baseline_s42", "Baseline (PV only)"), ("fieldaug_s42", "Field-style augmentation"),
         ("ft10_s42", "Fine-tune 10/class"), ("ft25_s42", "Fine-tune 25/class"),
         ("ft50_s42", "Fine-tune 50/class"), ("fieldaug_ft50_s42", "Augmentation + fine-tune 50/class")]
names = {int(r["class_id"]): r["plantdoc"] for r in csv.DictReader(open("../data_out/class_map.csv"))}
rows = []
for d, lab in ORDER:
    p = f"{R}/{d}/results.json"
    if not os.path.exists(p): continue
    r = json.load(open(p)); a, b = r["plantvillage_test"], r["plantdoc_test"]
    rows.append({"run": d, "condition": lab, "ft_images": r["config"].get("ft_images", 0),
                 "pv_macro_f1": a["macro_f1"], "pd_macro_f1": b["macro_f1"],
                 "pd_ci_lo": b["macro_f1_ci95"][0], "pd_ci_hi": b["macro_f1_ci95"][1],
                 "gap_pp": 100 * (a["macro_f1"] - b["macro_f1"]), "pd_accuracy": b["accuracy"],
                 "pd_top3": b["top3_accuracy"], "lat_med_ms": r["cpu_latency_ms"]["median"],
                 "lat_p95_ms": r["cpu_latency_ms"]["p95"]})
S = pd.DataFrame(rows)
base = S.loc[S.run == "baseline_s42", "pd_macro_f1"].iloc[0]
S["delta_vs_baseline_pp"] = 100 * (S.pd_macro_f1 - base)
S.to_csv("../runs/summary.csv", index=False); print(S.round(3).to_string(index=False))

# Fig 1: PlantDoc macro-F1 per condition with 95% cluster-bootstrap CI
fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(S))[::-1]
ax.errorbar(S.pd_macro_f1, y, xerr=[S.pd_macro_f1 - S.pd_ci_lo, S.pd_ci_hi - S.pd_macro_f1],
            fmt="o", color="#1f5f8b", ecolor="#7aa6c2", capsize=3)
ax.axvline(base, ls="--", lw=0.8, color="grey")
ax.set_yticks(y); ax.set_yticklabels(S.condition, fontsize=8)
ax.set_xlabel("Macro-F1 on PlantDoc field test (n = 445 crops), 95% CI"); ax.set_xlim(0, 1)
ax.set_title("Pilot: MobileNetV3-Large, seed 42", fontsize=9); fig.tight_layout()
fig.savefig(f"{F}/fig1_pd_macro_f1.png", dpi=200)

# Fig 2: per-class F1, PlantVillage vs PlantDoc, baseline
pv = pd.read_csv(f"{R}/baseline_s42/plantvillage_test_per_class.csv")
pdc = pd.read_csv(f"{R}/baseline_s42/plantdoc_test_per_class.csv")
o = pdc.sort_values("f1").label.values
fig, ax = plt.subplots(figsize=(7.2, 5.6)); yy = np.arange(len(o))
ax.barh(yy + 0.2, pv.set_index("label").loc[o, "f1"], 0.4, label="PlantVillage test (lab)", color="#9ab973")
ax.barh(yy - 0.2, pdc.set_index("label").loc[o, "f1"], 0.4, label="PlantDoc test (field)", color="#c0504d")
ax.set_yticks(yy); ax.set_yticklabels([f"{names[i]} (n={int(pdc.set_index('label').loc[i,'support'])})" for i in o], fontsize=6.5)
ax.set_xlabel("F1 per class (baseline model)"); ax.set_xlim(0, 1.0)
ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False); fig.tight_layout()
fig.savefig(f"{F}/fig2_per_class_f1.png", dpi=200)

# Fig 3: input/output examples (field test), baseline vs fine-tune 50
b = pd.read_csv(f"{R}/baseline_s42/plantdoc_test_predictions.csv")
f50 = pd.read_csv(f"{R}/ft50_s42/plantdoc_test_predictions.csv") if os.path.exists(f"{R}/ft50_s42") else b
m = b.merge(f50, on=["path", "label"], suffixes=("_b", "_f"))
rng = np.random.default_rng(7)
pick = pd.concat([m[(m.pred_b != m.label) & (m.pred_f == m.label)].sample(3, random_state=1),
                  m[(m.pred_b == m.label) & (m.pred_f == m.label)].sample(1, random_state=1),
                  m[(m.pred_b != m.label) & (m.pred_f != m.label)].sample(2, random_state=1)])
fig, axs = plt.subplots(1, 6, figsize=(10, 2.35))
for ax, (_, r) in zip(axs, pick.iterrows()):
    ax.imshow(Image.open(r.path).convert("RGB").resize((160, 160))); ax.axis("off")
    ax.set_title(f"True: {names[r.label]}\nBase: {names[r.pred_b]} ({r.conf_b:.2f})\nFT50: {names[r.pred_f]} ({r.conf_f:.2f})",
                 fontsize=5.6, loc="left")
fig.tight_layout(); fig.savefig(f"{F}/fig3_io_examples.png", dpi=220)
pick[["path", "label", "pred_b", "conf_b", "pred_f", "conf_f"]].to_csv("../runs/fig3_examples.csv", index=False)

# Selective prediction ('not sure') for each model on PlantDoc test - descriptive only
sel = []
for d, lab in ORDER:
    p = f"{R}/{d}/plantdoc_test_predictions.csv"
    if not os.path.exists(p): continue
    q = pd.read_csv(p); ok = q.pred == q.label
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ans = q.conf >= t
        sel.append({"condition": lab, "threshold": t, "coverage": ans.mean(),
                    "accuracy_answered": ok[ans].mean() if ans.any() else np.nan,
                    "errors_flagged_not_sure": (~ans & ~ok).sum() / max(1, (~ok).sum()),
                    "correct_withheld": (~ans & ok).sum() / max(1, ok.sum())})
pd.DataFrame(sel).to_csv("../runs/selective_prediction.csv", index=False)
print(pd.DataFrame(sel).round(3).to_string(index=False))
