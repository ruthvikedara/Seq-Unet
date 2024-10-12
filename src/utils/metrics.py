import numpy as np
import torch

def calculate_segmentation_metrics(predicted_mask, ground_truth_mask):
    predicted_mask_cpu = predicted_mask.cpu().numpy()
    predicted_mask = (predicted_mask_cpu >= 0.5).astype(np.uint8)
    ground_truth_mask = ground_truth_mask.unsqueeze(1).cpu().numpy()

    TP = np.sum((predicted_mask == 1) & (ground_truth_mask == 1))
    FP = np.sum((predicted_mask == 1) & (ground_truth_mask == 0))
    FN = np.sum((predicted_mask == 0) & (ground_truth_mask == 1))
    TN = np.sum((predicted_mask == 0) & (ground_truth_mask == 0))

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    dice = 2*TP / (2*TP + FN + FP) if 2*TP + FN + FP > 0 else 0.0

    return precision, recall, dice
