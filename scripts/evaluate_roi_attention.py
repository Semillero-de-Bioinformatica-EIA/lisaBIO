"""
Script de Evaluacion del Modelo Multimodal y Extraccion de Puntos Clave de Identificacion ROI.
Genera mapas de atencion ROI (Pat-XAI) e indicadores de desempeno.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.models.multimodal_fusion import MultimodalFusionNetwork
from dcis_biomarkers.pathology.mpm_dataset import MPMSequenceDataset
from dcis_biomarkers.multiomics.metabric_loader import METABRICDataset

RESULTS_DIR = Path("data/results")


def evaluate_and_extract_roi_attention():
    print("=== EVALUACION DE MODELO MULTIMODAL Y EXTRACCION DE ATENCION ROI ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultimodalFusionNetwork(
        omics_input_dim=1000,
        vision_embed_dim=512,
        omics_embed_dim=512,
        fused_dim=512,
        num_classes=3,
        use_monai=False
    ).to(device)
    
    checkpoint_path = RESULTS_DIR / "multimodal_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Pesos cargados desde: {checkpoint_path}")
    else:
        print("Aviso: No se encontro checkpoint guardado. Ejecutando evaluacion con modelo inicializado.")

    model.eval()
    
    mpm_dataset = MPMSequenceDataset(is_training=False)
    mpm_loader = DataLoader(mpm_dataset, batch_size=1, shuffle=False)
    
    metabric_dataset = METABRICDataset(num_top_genes=1000, is_training=False)
    metabric_loader = DataLoader(metabric_dataset, batch_size=1, shuffle=False)
    metabric_iter = iter(metabric_loader)
    
    attention_records = []
    
    with torch.no_grad():
        for pathology_batch in mpm_loader:
            case_id = pathology_batch["case_id"][0]
            seq_len = pathology_batch["seq_len"][0].item()
            diagnosis = pathology_batch["diagnosis"][0]
            seq_tensors = pathology_batch["sequence_tensor"].to(device)
            
            try:
                omics_batch = next(metabric_iter)
            except StopIteration:
                metabric_iter = iter(metabric_loader)
                omics_batch = next(metabric_iter)
                
            omic_tensors = omics_batch["omic_tensor"].to(device)
            
            outputs = model(seq_tensors, omic_tensors)
            logits = outputs["logits"]
            predicted_class = torch.argmax(logits, dim=1).item()
            
            roi_weights = outputs["roi_attention_weights"].squeeze().cpu().numpy()
            
            for r_idx in range(seq_len):
                w_val = float(roi_weights[r_idx]) if r_idx < len(roi_weights) else 0.0
                attention_records.append({
                    "case_id": case_id,
                    "roi_index": r_idx + 1,
                    "diagnosis": diagnosis,
                    "predicted_class": predicted_class,
                    "roi_attention_score": round(w_val, 4)
                })
                
    df_attn = pd.DataFrame(attention_records)
    output_csv = RESULTS_DIR / "roi_attention_scores.csv"
    df_attn.to_csv(output_csv, index=False)
    
    print("\n--- RESUMEN DE PUNTOS CLAVE DE ATENCION ROI (Pat-XAI) ---")
    print(df_attn.head(15))
    print(f"\nResultados de atencion exportados exitosamente a: {output_csv}")


if __name__ == "__main__":
    evaluate_and_extract_roi_attention()
