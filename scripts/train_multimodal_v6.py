"""
Multimodal Pipeline v6 - DCIS Progression Biomarkers.
Adjustments & Improvements:
  1. MPM Oversampling & Heavy Augmentations: Minorities (DCIS, IDC) oversampled in training batches
  2. Weighted Focal Loss (gamma=2.0, alpha=[3.0, 1.0, 3.0]) to penalize minority errors smoothly
  3. Non-Accuracy Evaluation Metrics: Balanced Accuracy and Macro F1-Score alongside Brier Score
  4. Differential Learning Rates: lr_vision = 1e-5, lr_head = 1e-4 with Gating Entropy penalty
"""

import os, sys, traceback, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.models.multimodal_fusion import MultimodalFusionNetwork
from dcis_biomarkers.pathology.mpm_dataset import MPMSequenceDataset
from dcis_biomarkers.multiomics.metabric_loader import METABRICDataset

RESULTS_DIR = Path("data/results")
HEATMAP_DIR = RESULTS_DIR / "roi_heatmaps"
LOG_FILE    = RESULTS_DIR / "training_v6_log.txt"

CLASS_NAMES = {0: "DCIS", 1: "DCISM", 2: "IDC"}


def log(msg: str, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss (Lin et al. 2017) con pesos alfa moderados [3.0, 1.0, 3.0]."""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_p = torch.log(target_probs + 1e-8)
        focal_weight = (1.0 - target_probs) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = alpha_weight * focal_weight
            
        loss = -focal_weight * log_p
        return loss.mean()


def compute_brier_score(probs: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> float:
    one_hot = np.eye(num_classes)[targets]
    bs = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(bs)


def cox_partial_loss(log_h: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order    = torch.argsort(times, descending=True)
    lh       = log_h[order].squeeze(-1)
    ev       = events[order]
    log_cs   = torch.logcumsumexp(lh, dim=0)
    loss     = -torch.mean(ev * (lh - log_cs))
    return torch.clamp(loss, -10.0, 10.0)


def save_roi_heatmap(case_id: str, attn_weights: np.ndarray, seq_len: int, diagnosis: str, pred: int, prob: np.ndarray):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(HEATMAP_DIR, exist_ok=True)
        roi_scores = attn_weights[:seq_len]
        fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.8), 3.5))
        
        norm_scores = (roi_scores - roi_scores.min()) / max(roi_scores.max() - roi_scores.min(), 1e-6)
        colors = [plt.cm.YlOrRd(0.3 + 0.7 * s) for s in norm_scores]
        
        ax.bar(
            range(1, seq_len + 1), roi_scores,
            color=colors, edgecolor="black", linewidth=0.8
        )
        ax.set_xlabel("ROI Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Attention Score (Sharpened)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"ROI Attention Heatmap v6 | {case_id}\n"
            f"GT: {diagnosis} | Pred: {CLASS_NAMES.get(pred, pred)} | "
            f"P(DCIS)={prob[0]:.2f} P(DCISM)={prob[1]:.2f} P(IDC)={prob[2]:.2f}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xticks(range(1, seq_len + 1))
        ax.set_ylim(0, max(roi_scores.max() * 1.3, 0.2))

        top_roi = int(np.argmax(roi_scores))
        ax.annotate(f"TOP ROI #{top_roi+1}\n({roi_scores[top_roi]:.3f})",
                    xy=(top_roi + 1, roi_scores[top_roi]),
                    xytext=(top_roi + 1, roi_scores[top_roi] * 1.15),
                    fontsize=8, ha="center", color="darkred", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))

        plt.tight_layout()
        save_path = HEATMAP_DIR / f"{case_id}_roi_heatmap_v6.png"
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        return str(save_path)
    except Exception as e:
        return f"[heatmap error: {e}]"


def run_pipeline_v6():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    f = open(LOG_FILE, "w")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[SISTEMA] Dispositivo : {device}", f)
        log(f"[SISTEMA] PyTorch     : {torch.__version__}", f)
        log("=" * 65, f)

        # ===== DATOS MPM CON OVERSAMPLING =====
        log("\n[1/5] Cargando Dataset MPM (Oversampling Clases Minoritarias)...", f)
        mpm_train = MPMSequenceDataset(is_training=True, oversample_minority=True)
        log(f"  >> Casos en batch de entrenamiento (Oversampled): {len(mpm_train)} muestras", f)

        log("\n[2/5] Cargando Dataset Omico METABRIC v6 (PAM50 + CNA + Clinica)...", f)
        meta_full   = METABRICDataset(num_top_genes=500, is_training=True)
        omic_dim    = meta_full.feature_dim
        log(f"  >> {len(meta_full)} pacientes | Feature dim = {omic_dim}", f)

        meta_loader = DataLoader(meta_full, batch_size=32, shuffle=True, drop_last=False)

        # ===== MODELO MULTIMODAL V6 =====
        log("\n[3/5] Construyendo Modelo Multimodal v6 (Weighted Focal Loss + Balanced Metrics)...", f)
        model = MultimodalFusionNetwork(
            omics_input_dim=omic_dim,
            vision_embed_dim=512,
            omics_embed_dim=512,
            fused_dim=512,
            num_classes=3,
            use_monai=False,
            freeze_backbone=False,
            temperature=0.5,
            gate_temperature=1.5
        ).to(device)

        # Learning Rate Diferencial
        vision_backbone_params = list(model.pathology_model.cnn_encoder.parameters())
        vision_backbone_ids = set(map(id, vision_backbone_params))
        rest_params = [p for p in model.parameters() if id(p) not in vision_backbone_ids]

        optimizer = optim.AdamW([
            {"params": vision_backbone_params, "lr": 1e-5, "weight_decay": 1e-3},
            {"params": rest_params,            "lr": 1e-4, "weight_decay": 1e-2}
        ])

        # WEIGHTED FOCAL LOSS [3.0, 1.0, 3.0]
        alpha_weights = torch.tensor([3.0, 1.0, 3.0], device=device)
        criterion_cls = WeightedFocalLoss(alpha=alpha_weights, gamma=2.0)

        EPOCHS = 25
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        log(f"\n[4/5] Entrenamiento v6 ({EPOCHS} epocas, Weighted Focal Loss [3.0, 1.0, 3.0])...", f)
        log("=" * 65, f)

        mpm_loader  = DataLoader(mpm_train, batch_size=3, shuffle=True, drop_last=False)
        meta_iter   = iter(meta_loader)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            ep_loss, ep_total = 0.0, 0
            vg_sum, og_sum, steps = 0.0, 0.0, 0
            all_probs, all_targets = [], []

            for step, pb in enumerate(mpm_loader):
                try:
                    seq = pb["sequence_tensor"].to(device)
                    lbl = pb["label"].to(device)

                    try:
                        ob = next(meta_iter)
                    except StopIteration:
                        meta_iter = iter(meta_loader)
                        ob = next(meta_iter)

                    omic  = ob["omic_tensor"].to(device)
                    rtim  = ob["rfs_time"].to(device)
                    rev   = ob["rfs_event"].to(device)
                    bs    = min(seq.size(0), omic.size(0))
                    seq, lbl, omic, rtim, rev = (
                        seq[:bs], lbl[:bs], omic[:bs], rtim[:bs], rev[:bs]
                    )

                    optimizer.zero_grad()
                    out = model(seq, omic)

                    loss_cls = criterion_cls(out["logits"], lbl)
                    
                    gw = out["gate_weights"]
                    entropy_penalty = 0.05 * torch.mean(torch.sum(gw * torch.log(gw + 1e-8), dim=-1))
                    
                    if rev.sum() > 0 and bs > 1:
                        loss_cox = cox_partial_loss(out["hazard_risk"], rtim, rev)
                        loss = loss_cls + 0.1 * loss_cox + entropy_penalty
                    else:
                        loss = loss_cls + entropy_penalty

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    probs = F.softmax(out["logits"], dim=-1).detach().cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                    all_probs.append(probs)
                    all_targets.append(lbl.cpu().numpy())

                    ep_total   += bs
                    ep_loss    += loss.item() * bs

                    gw_cpu = gw.detach().cpu().float()
                    vg_sum += gw_cpu[:, 0].mean().item()
                    og_sum += gw_cpu[:, 1].mean().item()
                    steps  += 1

                except Exception as e:
                    log(f"  [ERROR] Epoch {epoch} Step {step+1}: {e}", f)
                    traceback.print_exc(file=f)
                    continue

            scheduler.step()
            
            probs_concat  = np.concatenate(all_probs, axis=0)
            target_concat = np.concatenate(all_targets, axis=0)
            preds_concat  = np.argmax(probs_concat, axis=1)
            
            brier_score = compute_brier_score(probs_concat, target_concat, num_classes=3)
            bal_acc     = balanced_accuracy_score(target_concat, preds_concat)
            macro_f1    = f1_score(target_concat, preds_concat, average="macro")
            ep_l        = ep_loss / max(ep_total, 1)
            vg          = vg_sum  / max(steps, 1)
            og          = og_sum  / max(steps, 1)

            log(f"  Ep {epoch:02d}/{EPOCHS} | Loss {ep_l:.4f} | "
                f"BalAcc {bal_acc*100:.1f}% | MacroF1 {macro_f1:.3f} | "
                f"Brier {brier_score:.4f} | Gate[V={vg:.2f} O={og:.2f}]", f)

        # ===== GUARDAR MODELO V6 =====
        ckpt = RESULTS_DIR / "multimodal_model_v6.pth"
        torch.save(model.state_dict(), ckpt)
        log(f"\n[OK] Modelo v6 guardado : {ckpt}", f)

        # ===== EVALUACION SOBRE CASOS UNICOS SIN OVERSAMPLING =====
        log("\n[5/5] Evaluacion final sobre los 9 casos reales (sin oversampling)...", f)
        log("=" * 65, f)
        model.eval()
        mpm_eval  = MPMSequenceDataset(is_training=False, oversample_minority=False)
        mpm_eload = DataLoader(mpm_eval, batch_size=1, shuffle=False)
        me_iter   = iter(DataLoader(meta_full, batch_size=1, shuffle=False))

        records = []
        eval_probs, eval_targets = [], []
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total   = {0: 0, 1: 0, 2: 0}

        with torch.no_grad():
            for pb in mpm_eload:
                cid   = pb["case_id"][0]
                slen  = pb["seq_len"][0].item()
                diag  = pb["diagnosis"][0]
                lbl   = pb["label"].item()
                seq   = pb["sequence_tensor"].to(device)

                try:
                    ob = next(me_iter)
                except StopIteration:
                    me_iter = iter(DataLoader(meta_full, batch_size=1, shuffle=False))
                    ob = next(me_iter)

                omic  = ob["omic_tensor"].to(device)
                out   = model(seq, omic)
                prob  = F.softmax(out["logits"], dim=1).cpu().numpy()[0]
                pred  = int(np.argmax(prob))
                roi_w = out["roi_attention_weights"].squeeze().cpu().numpy()
                gw    = out["gate_weights"].squeeze().cpu().numpy()

                eval_probs.append(prob)
                eval_targets.append(lbl)

                class_total[lbl]   = class_total.get(lbl, 0) + 1
                class_correct[lbl] = class_correct.get(lbl, 0) + (1 if pred == lbl else 0)

                hpath = save_roi_heatmap(cid, roi_w, slen, diag, pred, prob)

                correct_str = "OK" if pred == lbl else "FAIL"
                log(f"  {cid:22s} | GT:{CLASS_NAMES[lbl]:6s} Pred:{CLASS_NAMES.get(pred,pred):6s} "
                    f"[{correct_str}] | "
                    f"Prob: DCIS={prob[0]:.2f} DCISM={prob[1]:.2f} IDC={prob[2]:.2f} | "
                    f"Gate[V={gw[0]:.2f} O={gw[1]:.2f}] | Heatmap: {Path(hpath).name}", f)

                for r in range(slen):
                    w = float(roi_w[r]) if r < len(roi_w) else 0.0
                    records.append({
                        "case_id": cid, "roi": r+1, "diagnosis": diag,
                        "gt_label": lbl, "pred_label": pred, "correct": pred == lbl,
                        "roi_attn": round(w, 5),
                        "gate_vision": round(float(gw[0]), 4),
                        "gate_omics":  round(float(gw[1]), 4),
                        "prob_dcis":   round(float(prob[0]), 4),
                        "prob_dcism":  round(float(prob[1]), 4),
                        "prob_idc":    round(float(prob[2]), 4)
                    })

        eval_probs_arr   = np.array(eval_probs)
        eval_targets_arr = np.array(eval_targets)
        eval_preds_arr   = np.argmax(eval_probs_arr, axis=1)

        final_brier = compute_brier_score(eval_probs_arr, eval_targets_arr, num_classes=3)
        final_bal_acc = balanced_accuracy_score(eval_targets_arr, eval_preds_arr)
        final_macro_f1 = f1_score(eval_targets_arr, eval_preds_arr, average="macro")

        log(f"\n  --- Evaluacion Final v6 (Metricas No-Accuracy) ---", f)
        log(f"    Balanced Accuracy : {final_bal_acc * 100:.1f}%", f)
        log(f"    Macro F1-Score    : {final_macro_f1:.4f}", f)
        log(f"    Brier Score       : {final_brier:.4f}", f)

        log("\n  --- Exactitud por Clase v6 ---", f)
        for cls_id, cls_name in CLASS_NAMES.items():
            ct = class_total.get(cls_id, 0)
            if ct > 0:
                acc = class_correct.get(cls_id, 0) / ct * 100
                log(f"    {cls_name:8s}: {class_correct.get(cls_id,0)}/{ct}  ({acc:.0f}%)", f)

        import pandas as pd
        df = pd.DataFrame(records)
        csv = RESULTS_DIR / "roi_attention_v6.csv"
        df.to_csv(csv, index=False)

        log("\n  --- Top 10 ROIs por Score de Atencion (v6 Sharpened) ---", f)
        log(df.sort_values("roi_attn", ascending=False).head(10).to_string(index=False), f)

        log("\n" + "=" * 65, f)
        log("[COMPLETADO] Pipeline v6 con Oversampling + Weighted Focal Loss + Macro F1", f)
        log(f"  Modelo    : {ckpt}", f)
        log(f"  CSV ROI   : {csv}", f)
        log(f"  Heatmaps  : {HEATMAP_DIR}/", f)
        log(f"  Log       : {LOG_FILE}", f)

    except Exception as e:
        log(f"\n[FATAL] {e}", f)
        traceback.print_exc(file=f)
    finally:
        f.close()
        print(f"\nLog guardado en: {LOG_FILE}")


if __name__ == "__main__":
    run_pipeline_v6()
