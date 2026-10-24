#!/bin/bash
# Pilot chain (seed 42). Baseline already run: python3 train_eval.py --condition baseline --out ../runs/baseline_s42
set -x
for K in 10 25 50; do
  python3 train_eval.py --condition ft$K --init ../runs/baseline_s42/model.pt --epochs 10 --lr 3e-4 --out ../runs/ft${K}_s42 > ../runs_ft$K.log 2>&1
done
python3 train_eval.py --condition fieldaug --out ../runs/fieldaug_s42 > ../runs_fieldaug.log 2>&1
python3 train_eval.py --condition ft50 --init ../runs/fieldaug_s42/model.pt --epochs 10 --lr 3e-4 --out ../runs/fieldaug_ft50_s42 > ../runs_fieldaug_ft50.log 2>&1
echo ALLDONE > ../runs_done.flag
