# Assignment 3 – pilot pipeline (Sergeyeva, Kuanyshbekova, SE-2403)

Code and real outputs for Part 7 of the report. Model checkpoints (`model.pt`, ~17 MB each) and image crops are not included; re-create them with the commands below.

Setup
- git clone https://github.com/spMohanty/PlantVillage-Dataset (uses raw/color, 27 class folders; commit 7f7ecc7)
- git clone https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset (commit 4730a23)
- git clone https://github.com/d-li14/mobilenetv3.pytorch  ->  export MBV3_DIR=<path to this clone>
- pip install torch torchvision scikit-learn pandas pillow imagehash matplotlib

Run (from code/)
1. python3 prepare_data.py --pv <PV>/raw/color --pd <PlantDoc-OD> --out ../data_out
2. python3 leakage_check.py ../data_out/pd_manifest.csv ../data_out/leakage_duplicates.csv <PlantDoc-OD>
3. python3 train_eval.py --condition baseline --out ../runs/baseline_s42
4. ./run_pilot.sh   (fine-tune 10/25/50, field-style augmentation, augmentation + fine-tune 50)
5. python3 analyse_pilot.py ; python3 error_analysis.py
6. python3 agroscan_infer.py ../runs/ft50_s42/model.pt 0.5 <photo.jpg>

Pilot environment: CPU, 2 threads, Python 3.13, torch 2.14.1, torchvision 0.29.1. On Colab GPU the full pilot takes a few minutes.
