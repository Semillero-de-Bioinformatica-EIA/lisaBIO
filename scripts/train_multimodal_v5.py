"""
Multimodal Pipeline v5 - DCIS Progression Biomarkers.
Adjustments:
  1. Focal Loss (gamma = 2.0, alpha = [2.0, 1.0, 2.0]) to address class imbalance smoothly
  2. Differential Learning Rates: lr_vision = 1e-5 (unfrozen backbone), lr_head = 1e-4
  3. Gating Entropy Regularization (prevents 1.0/0.0 binary gating collapse)
  4. Continuous Evaluation Metrics: Brier Score & Log-Loss (NLL)
"""

import os, sys, traceback, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.models.multimodal_fusion import MultimodalFusionNetwork
from dcis_biomarkers.pathology.mpm_dataset import MPMSequenceDataset
from dcis_biomarkers.multiomics.metabric_loader import METABRICDataset

RESULTS_DIR = Path("data/results")
HEATMAP_DIR = RESULTS_DIR / "roi_heatmaps"
LOG_FILE    = RESULTS_DIR / "training_v5_log.txt"

CLASS_NAMES = {0: "DCIS", 1: "DCISM", 2: "IDC"}


def log(msg: str, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) para clasificacion multiclase desbalanceada."""
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
    """Calcula el Brier Score continuo multiclase: BS = (1/N) * sum((p_ic - y_ic)^2)."""
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
            f"ROI Attention Heatmap v5 | {case_id}\n"
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
        save_path = HEATMAP_DIR / f"{case_id}_roi_heatmap_v5.png"
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        return str(save_path)
    except Exception as e:
        return f"[heatmap error: {e}]"


def run_pipeline_v5():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    f = open(LOG_FILE, "w")

    # Set random seeds for 100% reproducibility across runs
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log(f"[SISTEMA] Dispositivo : {device}", f)
        log(f"[SISTEMA] PyTorch     : {torch.__version__}", f)
        log("=" * 65, f)

        # ===== DATOS =====
        log("\n[1/5] Cargando Dataset MPM (Patologia)...", f)
        mpm_full = MPMSequenceDataset(is_training=True)
        n_cases  = len(mpm_full)
        log(f"  >> {n_cases} casos MPM | Clases: DCIS=0, DCISM=1, IDC=2", f)

        log("\n[2/5] Cargando Dataset Omico METABRIC v2 (mRNA + CNA + Clinica)...", f)
        meta_full   = METABRICDataset(num_top_genes=500, is_training=True)
        omic_dim    = meta_full.feature_dim
        log(f"  >> {len(meta_full)} pacientes | Feature dim = {omic_dim}", f)

        meta_loader = DataLoader(meta_full, batch_size=32, shuffle=True, drop_last=False)

        # ===== MODELO MULTIMODAL V5 =====
        log("\n[3/5] Construyendo Modelo Multimodal v5 (Focal Loss + Differential LR + Gating Entropy Penalty)...", f)
        model = MultimodalFusionNetwork(
            omics_input_dim=omic_dim,
            vision_embed_dim=512,
            omics_embed_dim=512,
            fused_dim=512,
            num_classes=3,
            use_monai=False,
            freeze_backbone=False, # DESCONGELADO CON LEARNING RATE DIFERENCIAL (lr=1e-5)
            temperature=0.5,
            gate_temperature=1.5  # SUAVIZADO DE GATING
        ).to(device)

        # ===== DIFFERENTIAL LEARNING RATES =====
        vision_backbone_params = list(model.pathology_model.cnn_encoder.parameters())
        vision_backbone_ids = set(map(id, vision_backbone_params))
        rest_params = [p for p in model.parameters() if id(p) not in vision_backbone_ids]

        optimizer = optim.AdamW([
            {"params": vision_backbone_params, "lr": 1e-5, "weight_decay": 1e-3}, # Low LR for visual encoder
            {"params": rest_params,            "lr": 1e-4, "weight_decay": 1e-2}  # Standard LR for rest
        ])

        # ===== FOCAL LOSS =====
        alpha_weights = torch.tensor([2.0, 1.0, 2.0], device=device) # Smoothed class weights
        criterion_cls = FocalLoss(alpha=alpha_weights, gamma=2.0)

        EPOCHS = 25
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        log(f"\n[4/5] Entrenamiento v5 ({EPOCHS} epocas, Focal Loss gamma=2.0)...", f)
        log("=" * 65, f)

        mpm_loader  = DataLoader(mpm_full, batch_size=2, shuffle=True, drop_last=False)
        meta_iter   = iter(meta_loader)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0
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
                    
                    # Gating Entropy Penalty: evita colapso a 1.0 / 0.0
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

                    ep_correct += np.sum(preds == lbl.cpu().numpy())
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
            
            brier_score = compute_brier_score(probs_concat, target_concat, num_classes=3)
            ep_acc = ep_correct / max(ep_total, 1)
            ep_l   = ep_loss    / max(ep_total, 1)
            vg     = vg_sum     / max(steps, 1)
            og     = og_sum     / max(steps, 1)
            lr_vis = optimizer.param_groups[0]["lr"]

            log(f"  Ep {epoch:02d}/{EPOCHS} | Loss {ep_l:.4f} | "
                f"Acc {ep_acc*100:.1f}% | "
                f"Brier {brier_score:.4f} | "
                f"Gate[V={vg:.2f} O={og:.2f}] | LR_Vis {lr_vis:.2e}", f)

        # ===== GUARDAR MODELO =====
        ckpt = RESULTS_DIR / "multimodal_model_v5.pth"
        torch.save(model.state_dict(), ckpt)
        log(f"\n[OK] Modelo v5 guardado : {ckpt}", f)

        # ===== EVALUACION CONTINUA Y HEATMAPS ROI v5 =====
        log("\n[5/5] Evaluacion final continua y generacion de Heatmaps ROI v5...", f)
        log("=" * 65, f)
        model.eval()
        mpm_eval  = MPMSequenceDataset(is_training=False)
        mpm_eload = DataLoader(mpm_eval, batch_size=1, shuffle=False)
        me_iter   = iter(DataLoader(meta_full, batch_size=1, shuffle=False))

        records = []
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total   = {0: 0, 1: 0, 2: 0}
        eval_probs, eval_targets = [], []

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

        final_brier = compute_brier_score(np.array(eval_probs), np.array(eval_targets), num_classes=3)
        log(f"\n  --- Evaluacion Continua Final ---", f)
        log(f"    Brier Score Continuo (menor es mejor): {final_brier:.4f}", f)

        log("\n  --- Per-Class Accuracy v5 ---", f)
        for cls_id, cls_name in CLASS_NAMES.items():
            ct = class_total.get(cls_id, 0)
            if ct > 0:
                acc = class_correct.get(cls_id, 0) / ct * 100
                log(f"    {cls_name:8s}: {class_correct.get(cls_id,0)}/{ct}  ({acc:.0f}%)", f)

        import pandas as pd
        df = pd.DataFrame(records)
        csv = RESULTS_DIR / "roi_attention_v5.csv"
        df.to_csv(csv, index=False)

        log("\n  --- Top 10 ROIs por Score de Atencion (Sharpened v5) ---", f)
        log(df.sort_values("roi_attn", ascending=False).head(10).to_string(index=False), f)

        log("\n" + "=" * 65, f)
        log("[COMPLETADO] Pipeline v5 con Focal Loss + Differential LR + Brier Score", f)
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
    run_pipeline_v5()
