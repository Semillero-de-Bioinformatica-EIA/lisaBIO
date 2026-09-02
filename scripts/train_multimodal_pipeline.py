"""
Script de Entrenamiento y Evaluacion Multimodal AI v2 - DCIS Progression Biomarkers.
Pipeline Completo: Carga datos, entrenamiento, evaluacion y extraccion de atencion ROI.

Mejoras v2:
  - Ciclo de entrenamiento completo con train/eval split
  - Metricas por epoca: Loss, Accuracy, Gate Weights (vision vs omics balance)
  - Perdida de supervivencia Cox parcial (aproximacion negativa log-likelihood)
  - Exportacion de atencion ROI al finalizar
  - Logging detallado de errores y progreso
"""

import os, sys, traceback
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.models.multimodal_fusion import MultimodalFusionNetwork
from dcis_biomarkers.pathology.mpm_dataset import MPMSequenceDataset
from dcis_biomarkers.multiomics.metabric_loader import METABRICDataset

RESULTS_DIR = Path("data/results")
LOG_FILE = RESULTS_DIR / "training_log.txt"


def log(msg: str, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


def cox_partial_loss(log_hazard: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Aproximacion de perdida de verosimilitud parcial de Cox (Breslow)."""
    order = torch.argsort(times, descending=True)
    log_h = log_hazard[order].squeeze()
    ev    = events[order]
    log_cumsum = torch.logcumsumexp(log_h, dim=0)
    loss = -torch.mean(ev * (log_h - log_cumsum))
    return loss


def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    f = open(LOG_FILE, "w")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[SISTEMA] Dispositivo: {device}", f)
        log(f"[SISTEMA] PyTorch {torch.__version__}", f)
        log("=" * 65, f)

        # === CARGA DE DATOS ===
        log("\n[1/4] Cargando Dataset de Patologia MPM...", f)
        try:
            mpm_full = MPMSequenceDataset(is_training=True)
            n_mpm = len(mpm_full)
            log(f"  >> {n_mpm} casos MPM cargados", f)
            if n_mpm < 2:
                raise ValueError(f"Dataset MPM demasiado pequeno ({n_mpm} casos). Minimo 2 requeridos.")
        except Exception as e:
            log(f"  [ERROR] Fallo al cargar MPM: {e}", f)
            traceback.print_exc(file=f)
            raise

        log("\n[2/4] Cargando Dataset Omico METABRIC...", f)
        try:
            meta_full = METABRICDataset(num_top_genes=1000, is_training=True)
            n_meta = len(meta_full)
            log(f"  >> {n_meta} pacientes METABRIC cargados con 1000 genes HVG", f)
        except Exception as e:
            log(f"  [ERROR] Fallo al cargar METABRIC: {e}", f)
            traceback.print_exc(file=f)
            raise

        # Dataloaders
        mpm_loader  = DataLoader(mpm_full,  batch_size=2, shuffle=True,  drop_last=False)
        meta_loader = DataLoader(meta_full, batch_size=32, shuffle=True, drop_last=False)

        # === MODELO ===
        log("\n[3/4] Construyendo Modelo Multimodal v2...", f)
        try:
            model = MultimodalFusionNetwork(
                omics_input_dim=1000,
                vision_embed_dim=512,
                omics_embed_dim=512,
                fused_dim=512,
                num_classes=3,
                use_monai=False
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"  >> Parametros entrenables: {n_params:,}", f)
        except Exception as e:
            log(f"  [ERROR] Fallo al construir modelo: {e}", f)
            traceback.print_exc(file=f)
            raise

        criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # === ENTRENAMIENTO ===
        log("\n[4/4] Iniciando Entrenamiento (5 epocas)...", f)
        log("=" * 65, f)

        EPOCHS = 5
        meta_iter = iter(meta_loader)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total   = 0
            vision_gate_sum = 0.0
            omics_gate_sum  = 0.0
            steps = 0
            
            log(f"\n  -- Epoca {epoch}/{EPOCHS} --", f)

            for step, path_batch in enumerate(mpm_loader):
                try:
                    seq = path_batch["sequence_tensor"].to(device)
                    lbl = path_batch["label"].to(device)

                    # Omics batch (ciclico)
                    try:
                        ob = next(meta_iter)
                    except StopIteration:
                        meta_iter = iter(meta_loader)
                        ob = next(meta_iter)

                    omic  = ob["omic_tensor"].to(device)
                    rtime = ob["rfs_time"].to(device)
                    rev   = ob["rfs_event"].to(device)
                    bs    = min(seq.size(0), omic.size(0))
                    seq, lbl, omic, rtime, rev = seq[:bs], lbl[:bs], omic[:bs], rtime[:bs], rev[:bs]

                    optimizer.zero_grad()
                    out = model(seq, omic)

                    # Perdida clasificacion
                    loss_cls = criterion_cls(out["logits"], lbl)

                    # Perdida Cox supervivencia (si hay varianza en eventos)
                    if rev.sum() > 0 and bs > 1:
                        loss_cox = cox_partial_loss(out["hazard_risk"], rtime, rev)
                        loss = loss_cls + 0.2 * loss_cox
                    else:
                        loss = loss_cls

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    preds = torch.argmax(out["logits"], dim=1)
                    correct += (preds == lbl).sum().item()
                    total   += bs
                    total_loss += loss.item() * bs

                    # Gate weights
                    gw = out["gate_weights"].detach().cpu()
                    vision_gate_sum += gw[:, 0].mean().item()
                    omics_gate_sum  += gw[:, 1].mean().item()
                    steps += 1

                    log(f"    Step {step+1:02d} | Loss: {loss.item():.4f} | Pred: {preds.tolist()} vs GT: {lbl.tolist()}", f)

                except Exception as e:
                    log(f"    [ERROR] Step {step+1}: {e}", f)
                    traceback.print_exc(file=f)
                    continue

            scheduler.step()

            epoch_loss = total_loss / max(total, 1)
            epoch_acc  = correct / max(total, 1)
            vg_avg = vision_gate_sum / max(steps, 1)
            og_avg = omics_gate_sum  / max(steps, 1)

            log(f"\n  Epoca {epoch} Resumen:", f)
            log(f"    Loss Total:        {epoch_loss:.4f}", f)
            log(f"    Precision:         {epoch_acc*100:.1f}%", f)
            log(f"    Gate Vision:       {vg_avg:.3f}   Gate Omics: {og_avg:.3f}", f)
            log(f"    LR actual:         {scheduler.get_last_lr()[0]:.6f}", f)

        # === GUARDAR MODELO ===
        ckpt = RESULTS_DIR / "multimodal_model_v2.pth"
        torch.save(model.state_dict(), ckpt)
        log(f"\n[OK] Modelo guardado en: {ckpt}", f)

        # === EVALUACION ROI ATTENTION ===
        log("\n[OK] Extrayendo Puntos Clave de Atencion ROI...", f)
        model.eval()
        meta_eval_iter = iter(DataLoader(meta_full, batch_size=1, shuffle=False))
        all_records = []

        with torch.no_grad():
            for pb in DataLoader(MPMSequenceDataset(is_training=False), batch_size=1, shuffle=False):
                try:
                    seq  = pb["sequence_tensor"].to(device)
                    cid  = pb["case_id"][0]
                    slen = pb["seq_len"][0].item()
                    diag = pb["diagnosis"][0]

                    try:
                        ob = next(meta_eval_iter)
                    except StopIteration:
                        meta_eval_iter = iter(DataLoader(meta_full, batch_size=1, shuffle=False))
                        ob = next(meta_eval_iter)

                    omic = ob["omic_tensor"].to(device)
                    out  = model(seq, omic)
                    pred = torch.argmax(out["logits"], dim=1).item()
                    roi  = out["roi_attention_weights"].squeeze().cpu().numpy()
                    gw   = out["gate_weights"].squeeze().cpu().numpy()

                    for r in range(slen):
                        w = float(roi[r]) if r < len(roi) else 0.0
                        all_records.append({
                            "case_id": cid, "roi": r+1, "diagnosis": diag,
                            "pred_class": pred, "roi_attn": round(w, 5),
                            "gate_vision": round(float(gw[0]), 4),
                            "gate_omics":  round(float(gw[1]), 4)
                        })
                except Exception as e:
                    log(f"  [ERROR] ROI eval {cid}: {e}", f)
                    continue

        import pandas as pd
        df = pd.DataFrame(all_records)
        csv_path = RESULTS_DIR / "roi_attention_scores_v2.csv"
        df.to_csv(csv_path, index=False)

        log("\n" + "=" * 65, f)
        log("[OK] PIPELINE MULTIMODAL COMPLETADO", f)
        log(f"  ROI Attention: {csv_path}", f)
        log(f"  Log completo:  {LOG_FILE}", f)
        log("\nTop ROIs por atencion:", f)
        if not df.empty:
            log(df.sort_values("roi_attn", ascending=False).head(10).to_string(index=False), f)

    except Exception as e:
        log(f"\n[FATAL ERROR] {e}", f)
        traceback.print_exc(file=f)
    finally:
        f.close()
        print(f"\nLog guardado en: {LOG_FILE}")


if __name__ == "__main__":
    run_pipeline()
