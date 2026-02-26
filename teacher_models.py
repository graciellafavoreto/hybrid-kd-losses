"""
Teacher Training for Knowledge Distillation
────────────────────────────────────────────
Models: ResNet-50, DenseNet-121, ViT-Small, Swin-Base, MaxViT-Tiny
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import timm
import random
from config import BASE_PATH
from utils import NoduleDataset, assign_labels, evaluate, patient_split

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
EPOCHS = 50
LR = 1e-4
PATIENCE = 8
BATCH = 16
CKPT_DIR = 'checkpoints_teachers_equals'     

TEACHERS = {
    'resnet50': 'resnet50',
    'densenet121': 'densenet121',
    'vit_small': 'vit_small_patch16_224',
    'swin_transformer': 'swin_base_patch4_window7_224',
    'maxvit_tiny': 'maxvit_tiny_tf_224',
}


# Model Builder
def build_teacher(timm_name: str) -> nn.Module:
    return timm.create_model(
        timm_name, pretrained=True,
        in_chans=3, num_classes=2, drop_rate=0.3,
    )

# Training
def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        total += loss.item()
        
    return total / len(loader)

# Main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data and assign labels
    df = pd.read_csv(Path(BASE_PATH) / 'nodules_metadata.csv')
    df = assign_labels(df)[lambda d: d['label'] != -1].reset_index(drop=True)

    # Split patients into train/val/test
    df_train, df_val, df_test = patient_split(df, test_size=1/7, val_size=1/6)
    
    print(
        f"Split: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}\n"
    )

    print(
        f"Patients — Train: {df_train['patient_id'].nunique()} | "
        f"Val: {df_val['patient_id'].nunique()} | "
        f"Test: {df_test['patient_id'].nunique()}\n"
    )

    n_b = (df_train['label'] == 0).sum()
    n_m = (df_train['label'] == 1).sum()
    
    cw = torch.tensor(
        [len(df_train) / (2 * n_b), len(df_train) / (2 * n_m)],
        dtype=torch.float32,
    ).to(device)

    nw = min(8, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        NoduleDataset(df_train, BASE_PATH, augment=True),
        BATCH, shuffle=True, num_workers=nw, pin_memory=True
    )

    val_loader = DataLoader(
        NoduleDataset(df_val, BASE_PATH, augment=False),
        BATCH, shuffle=False, num_workers=nw, pin_memory=True
    )

    test_loader = DataLoader(
        NoduleDataset(df_test, BASE_PATH, augment=False),
        BATCH, shuffle=False, num_workers=nw, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss(weight=cw)  
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Phase 1: Train Teachers (Selection by Validation AUC)
    val_results = {}

    for model_name, timm_name in TEACHERS.items():
        print("=" * 70)
        print(f"Teacher: {model_name.upper()}")
        print("=" * 70)

        try:
            model = build_teacher(timm_name).to(device)
        except Exception as e:
            print(f"ERROR loading  {model_name}: {e}\n")
            continue

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=0.05
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        
        ckpt_path = f'{CKPT_DIR}/{model_name}.pth'
        best_auc = 0.0
        patience_count = 0

        print(f"{'Ep':>3} {'Loss':>7}  {'ValAcc':>7} {'ValAUC':>7} {'ValSens':>7} {'ValSpec':>7}")
        print("-" * 65)

        for epoch in range(EPOCHS):
            loss = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            vm = evaluate(model, val_loader, device)
            scheduler.step()

            saved = ''
            
            if vm['auc'] > best_auc:
                best_auc = vm['auc']
                patience_count = 0
                torch.save(model.state_dict(), ckpt_path)
                saved = ' *'
            else:
                patience_count += 1

            print(f"{epoch+1:>3} {loss:>7.4f}  "
                  f"{vm['acc']:>7.4f} {vm['auc']:>7.4f} "
                  f"{vm['sens']:>7.4f} {vm['spec']:>7.4f}{saved}")

            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)
        )
        
        vm_f = evaluate(model, val_loader, device)
        
        val_results[model_name] = {'model': model, 'ckpt': ckpt_path, **vm_f}
        
        print(f"\nBest Validation AUC: {vm_f['auc']:.4f}\n")

    # Phase 2: Final Test Evaluation 
    print("\n" + "=" * 70)
    print("FINAL EVALUATION — Test Set")
    print("=" * 70)
    print(f"{'Teacher':<20} {'Acc':>7} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7}")
    print("-" * 70)

    rows = []
    
    for model_name, res in val_results.items():
        tm = evaluate(res['model'], test_loader, device)
        tn, fp, fn, tp = tm['confusion']
        
        print(f"{model_name:<20} "
              f"{tm['acc']:>7.4f} {tm['auc']:>7.4f} "
              f"{tm['sens']:>7.4f} {tm['spec']:>7.4f} {tm['f1']:>7.4f}")

        print(f"{'':20} Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        
        rows.append({
            'model': model_name,
            'val_auc': round(res['auc'], 4),
            'test_accuracy': round(tm['acc'], 4),
            'test_auc': round(tm['auc'], 4),
            'test_sensitivity': round(tm['sens'], 4),
            'test_specificity': round(tm['spec'], 4),
            'test_f1': round(tm['f1'], 4),
        })

    pd.DataFrame(rows).to_csv(f'{CKPT_DIR}/teacher_results.csv', index=False)
    
    print(f"\nSaved: {CKPT_DIR}/teacher_results.csv")
    print(f"Checkpoints saved in: {CKPT_DIR}/<model>.pth")

if __name__ == '__main__':
    main()
