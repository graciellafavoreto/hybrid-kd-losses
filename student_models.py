"""
Knowledge Distillation — Students: EfficientNet-B0 and MobileNetV2
─────────────────────────────────────────────────────────────────
4 distillation functions x 3 temperatures x 5 teachers x 2 students = 120 experiments

  D_L   (classic)        
  D_L1  (KL+MSE)        
  D_L2  (KL+Focal)       
  D_L3  (KL+MSE+Focal)   
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import timm
import random
from config import *
from utils import NoduleDataset, assign_labels, evaluate, patient_split

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Models


def build_teacher(name: str, device) -> nn.Module:
    model = timm.create_model(
        TEACHERS[name],
        pretrained=False,
        in_chans=3,
        num_classes=2
    )
    ckpt = torch.load(
        f'{TEACHER_DIR}/{name}.pth',
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(ckpt)
    model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def build_student(name: str, device) -> nn.Module:
    return timm.create_model(
        STUDENTS[name],
        pretrained=True,
        in_chans=3,
        num_classes=2,
        drop_rate=0.3
    ).to(device)


# Distillation loss
def focal_loss_soft(p_s: torch.Tensor, p_t: torch.Tensor, gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    """Focal loss between soft probability distributions"""
    return (-((1 - p_s) ** gamma) * p_t * torch.log(p_s + 1e-8)).sum(dim=1).mean()


def distillation_loss(s_logits, t_logits, y_hard, cw,
                      loss_variant: str, T: int,
                      alpha: float = ALPHA) -> torch.Tensor:

    CE = nn.CrossEntropyLoss(weight=cw)(s_logits, y_hard)

    # Temperature-scaled soft distributions
    s_soft = F.softmax(s_logits / T, dim=1)
    t_soft = F.softmax(t_logits / T, dim=1)

    kl = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        t_soft,
        reduction='batchmean'
    )

    if loss_variant == 'classic':
        D = kl
    elif loss_variant == 'kl_mse':
        D = kl + F.mse_loss(s_soft, t_soft)
    elif loss_variant == 'kl_focal':
        D = kl + focal_loss_soft(s_soft, t_soft)
    elif loss_variant == 'kl_mse_focal':
        D = kl + F.mse_loss(s_soft, t_soft) + focal_loss_soft(s_soft, t_soft)
    else:
        raise ValueError(f"Unknown loss variant: {loss_variant}")

    return (1 - alpha) * CE + alpha * (T**2) * D


# Training
def train_epoch(student, teacher, loader, optimizer, cw,
                loss_variant, T, device) -> float:

    student.train()
    total = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        s_logits = student(x)

        with torch.no_grad():
            t_logits = teacher(x)

        loss = distillation_loss(s_logits, t_logits, y, cw, loss_variant, T)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


# Main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_exp = len(TEACHERS) * len(STUDENTS) * \
        len(TEMPERATURES) * len(LOSS_VARIANTS)

    print(f"Device: {device}")
    print(
        f"Temperatures: {TEMPERATURES} | Alpha: {ALPHA} | Focal γ: {FOCAL_GAMMA}")
    print(f"Total experiments: {total_exp}\n")
    print(
        f"Fixed hyperparameters: LR={LR} | Epochs={EPOCHS} | Batch={BATCH} | Patience={PATIENCE}\n")

    # Data — same split used in train_teachers.py
    df = pd.read_csv(Path(BASE_PATH) / 'nodules_metadata.csv')
    df = assign_labels(df)[lambda d: d['label'] != -1].reset_index(drop=True)

    df_train, df_val, df_test = patient_split(df, test_size=1/7, val_size=1/6)
    print(
        f"Split: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}\n")

    n_b = (df_train['label'] == 0).sum()
    n_m = (df_train['label'] == 1).sum()

    cw = torch.tensor(
        [len(df_train) / (2 * n_b), len(df_train) / (2 * n_m)],
        dtype=torch.float32,
    ).to(device)

    nw = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        NoduleDataset(df_train, BASE_PATH, augment=True),
        BATCH,
        shuffle=True,
        num_workers=nw,
        pin_memory=True
    )

    val_loader = DataLoader(
        NoduleDataset(df_val, BASE_PATH, augment=False),
        BATCH,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )

    test_loader = DataLoader(
        NoduleDataset(df_test, BASE_PATH, augment=False),
        BATCH,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    csv_path = f'{CKPT_DIR}/all_val_results.csv'
    done_keys: set[tuple] = set()

    if os.path.exists(csv_path):
        df_prev = pd.read_csv(csv_path)
        all_results = df_prev.to_dict('records')
        for row in all_results:
            done_keys.add(
                (
                    row['teacher'],
                    row['student'],
                    int(row['tau']),
                    row['loss_variant']
                )
            )
        print(f"[RESUME] {len(done_keys)} experiments already completed")
    else:
        all_results = []

    exp_count = 0

    # Phase 1: 120 experiments, checkpoint saved by val_auc
    for teacher_name in TEACHERS:
        ckpt_t = f'{TEACHER_DIR}/{teacher_name}.pth'
        if not os.path.exists(ckpt_t):
            print(f"[SKIP] {ckpt_t} not found")
            continue

        teacher = build_teacher(teacher_name, device)

        for student_name in STUDENTS:
            for T in TEMPERATURES:
                print("=" * 85)
                print(
                    f"Teacher={teacher_name.upper()} | Student={student_name} | t={T}")
                print("=" * 85)

                for loss_variant in LOSS_VARIANTS:
                    vlabel = VARIANT_LABEL[loss_variant]
                    ckpt_path = f'{CKPT_DIR}/{teacher_name}_{student_name}_T{T}_{loss_variant}.pth'

                    exp_count += 1
                    skip_key = (teacher_name, student_name, T, loss_variant)
                    if skip_key in done_keys:
                        print(
                            f"[{exp_count}/{total_exp}] {vlabel} → ALREADY COMPLETED, skipping.")
                        continue

                    print(f"\n[{exp_count}/{total_exp}] Loss: {vlabel}")
                    print(f"{'Ep':>3} {'Loss':>8}  {'ValAcc':>7} {'ValAUC':>7} {'ValSens':>7} {'ValSpec':>7}")
                    print(f"{'-' * 65}")

                    student = build_student(student_name, device)
                    
                    optimizer = torch.optim.AdamW(
                        student.parameters(), lr=LR, weight_decay=0.05
                    )
                    
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS)

                    best_auc = 0.0
                    patience_count = 0

                    for epoch in range(EPOCHS):
                        loss = train_epoch(
                            student, teacher, train_loader,
                            optimizer, cw, loss_variant, T, device
                        )
                        
                        vm = evaluate(student, val_loader, device)
                        scheduler.step()

                        saved = ''
                        if vm['auc'] > best_auc:
                            best_auc = vm['auc']
                            patience_count = 0
                            torch.save(student.state_dict(), ckpt_path)
                            saved = ' *'
                        else:
                            patience_count += 1

                        print(f"  {epoch+1:>3} {loss:>8.4f}  "
                              f"{vm['acc']:>7.4f} {vm['auc']:>7.4f} "
                              f"{vm['sens']:>7.4f} {vm['spec']:>7.4f}{saved}")

                        if patience_count >= PATIENCE:
                            print(f"Early stopping at the epoch {epoch+1}")
                            break

                    # Final evaluation of the experiment
                    student.load_state_dict(
                        torch.load(ckpt_path, map_location=device, weights_only=False)
                    )
                    
                    vm_f = evaluate(student, val_loader, device)
                    
                    print(f"\n  BEST Val AUC: {vm_f['auc']:.4f}\n")

                    all_results.append({
                        'teacher': teacher_name,
                        'student': student_name,
                        'tau': T,
                        'loss': vlabel,
                        'loss_variant': loss_variant,
                        'ckpt_path': ckpt_path,
                        'val_acc': round(vm_f['acc'],  4),
                        'val_auc': round(vm_f['auc'],  4),
                        'val_sens': round(vm_f['sens'], 4),
                        'val_spec': round(vm_f['spec'], 4),
                        'val_f1': round(vm_f['f1'],   4),
                    })

                    pd.DataFrame(all_results).to_csv(
                        f'{CKPT_DIR}/all_val_results.csv', index=False)
                    
    print("\nSaved:")
    print(f"{CKPT_DIR}/all_val_results.csv")

    df_res = pd.DataFrame(all_results)
    if df_res.empty:
        print(
            "\n[ERROR] No experiments were executed. Check teacher checkpoints")
        return

    # Phase 2: Select best configuration by Val AUC
    print("\n" + "=" * 85)
    print("FINAL SELECTION — Based on Validation AUC")
    print("=" * 85)

    final_results = []

    for s_name in STUDENTS:
        sub = df_res[df_res['student'] == s_name]
        if sub.empty:
            continue

        best_row = sub.loc[sub['val_auc'].idxmax()]  # selection based on val_auc
        print(f"\n[{s_name}] Best configuration (val_auc={best_row['val_auc']:.4f}):")
        print(
            f"Teacher={best_row['teacher']} | Loss={best_row['loss']} | tau={int(best_row['tau'])}"
        )

        student = timm.create_model(
            STUDENTS[s_name], 
            pretrained=False,
            in_chans=3, 
            num_classes=2, 
            drop_rate=0.3,
        ).to(device)
        
        student.load_state_dict(
            torch.load(
                best_row['ckpt_path'],
                map_location=device,
                weights_only=False
            )
        )

        tm = evaluate(student, test_loader, device)
        tn, fp, fn, tp = tm['confusion']
        
        print(
            f"  Test → Acc={tm['acc']:.4f} "
            f"AUC={tm['auc']:.4f} "
            f"Sensitivity={tm['sens']:.4f} "
            f"Specificity={tm['spec']:.4f}"
        )
         
        print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

        final_results.append({
            'student': s_name,
            'teacher': best_row['teacher'],
            'tau': int(best_row['tau']),
            'loss': best_row['loss'],
            'val_auc': best_row['val_auc'],
            'test_acc': round(tm['acc'],  4),
            'test_auc': round(tm['auc'],  4),
            'test_sens': round(tm['sens'], 4),
            'test_spec': round(tm['spec'], 4),
            'test_f1': round(tm['f1'],   4),
        })

    pd.DataFrame(final_results).to_csv(
        f'{CKPT_DIR}/final_test_results.csv', 
        index=False
    )

if __name__ == '__main__':
    main()
