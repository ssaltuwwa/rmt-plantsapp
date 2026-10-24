"""O4 core function of the AgroScan Telegram bot: photo -> top-3 diagnoses with confidence,
or "not sure" when the top-1 confidence is below a threshold. The Telegram layer (python-telegram-bot
handler that downloads the photo and calls diagnose()) is a thin wrapper and is not part of this pilot.
Usage: python3 agroscan_infer.py MODEL.pt THRESHOLD img1.jpg [img2.jpg ...]"""
import sys, time, csv, torch
from PIL import Image
from train_eval import build_model, tfs

CLASSES = [r["plantvillage"] for r in csv.DictReader(open("../data_out/class_map.csv"))]


def load(path):
    m = build_model(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); return m


@torch.no_grad()
def diagnose(m, img_path, threshold=0.5, size=224):
    t0 = time.perf_counter()
    x = tfs("eval", size)(Image.open(img_path).convert("RGB")).unsqueeze(0)
    p = torch.softmax(m(x), 1)[0]
    conf, idx = p.topk(3)
    top3 = [(CLASSES[i].replace("___", " - ").replace("_", " "), round(float(c), 3)) for c, i in zip(conf, idx)]
    status = "ok" if float(conf[0]) >= threshold else "not sure"
    return {"status": status, "top3": top3, "ms": round(1000 * (time.perf_counter() - t0), 1)}


if __name__ == "__main__":
    torch.set_num_threads(2)
    m = load(sys.argv[1]); thr = float(sys.argv[2])
    for f in sys.argv[3:]:
        print(f.split("/")[-1][:50], "->", diagnose(m, f, thr))
