# Add Optuna tuning or fine-tuning logic for SAM2 here
import os
import sys
print(os.getcwd())
# os.chdir('..')
print(os.getcwd())
print(os.listdir())
sys.path.insert(0, 'object_delineation')
sys.path.insert(0, '../sam2')
from eval_sam2_raster import eval_masks
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import optuna
import torch
import pandas as pd
import numpy as np

main_dir = '/rhome/adist003/bigdata/eval_sam2'

def freeze_model_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'output' in name or 'head' in name:
            param.requires_grad = True

def objective(trial):
    confidence_threshold = trial.suggest_uniform('confidence_threshold', 0.5, 0.9)
    points_per_side = trial.suggest_categorical('points_per_side', [16, 32, 64])
    points_per_batch = trial.suggest_categorical('points_per_batch', [32, 64, 128])
    pred_iou_thresh = trial.suggest_uniform('pred_iou_thresh', 0.5, 0.95)
    stability_score_thresh = trial.suggest_uniform('stability_score_thresh', 0.8, 1.0)
    stability_score_offset = trial.suggest_uniform('stability_score_offset', 0.5, 1.5)
    crop_n_layers = trial.suggest_int('crop_n_layers', 0, 3)
    crop_n_points_downscale_factor = trial.suggest_uniform('crop_n_points_downscale_factor', 0.5, 2.0)
    box_nms_thresh = trial.suggest_uniform('box_nms_thresh', 0.5, 0.9)
    min_mask_region_area = trial.suggest_int('min_mask_region_area', 0, 500)
    use_m2m = trial.suggest_categorical('use_m2m', [True, False])

    model_cfg_path = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    checkpoint_path = 'sam2.1_hiera_large.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam2 = build_sam2(model_cfg_path, checkpoint_path, device=device, apply_postprocessing=False)
    freeze_model_layers(sam2)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        box_nms_thresh=box_nms_thresh,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m
    )

    eval_stats = eval_masks_parallel(main_dir, print_df=False)
    mean_recall = eval_stats['recall'].mean()
    mean_precision = eval_stats['precision'].mean()
    mean_f1_score = eval_stats['f1_score'].mean()

    results_row = {
        'confidence_threshold': confidence_threshold,
        'points_per_side': points_per_side,
        'points_per_batch': points_per_batch,
        'pred_iou_thresh': pred_iou_thresh,
        'stability_score_thresh': stability_score_thresh,
        'stability_score_offset': stability_score_offset,
        'crop_n_layers': crop_n_layers,
        'crop_n_points_downscale_factor': crop_n_points_downscale_factor,
        'box_nms_thresh': box_nms_thresh,
        'min_mask_region_area': min_mask_region_area,
        'use_m2m': use_m2m,
        'recall': mean_recall,
        'precision': mean_precision,
        'f1_score': mean_f1_score
    }

    results_df = pd.DataFrame([results_row])
    if not os.path.exists('hyperparameter_results_sam2_real.csv'):
        results_df.to_csv('hyperparameter_results_sam2_real.csv', index=False)
    else:
        results_df.to_csv('hyperparameter_results_sam2_real.csv', mode='a', header=False, index=False)

    return mean_recall

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters: ', study.best_params)
print('Best recall: ', study.best_value)