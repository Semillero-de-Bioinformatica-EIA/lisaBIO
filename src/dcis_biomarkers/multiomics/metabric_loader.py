"""
METABRIC Multi-omics Data Loader - v6 (Deterministic PAM50 + Driver Biomarkers).
Mejoras:
  - Garantiza la inclusion determinista de los 54 genes clave PAM50 y Drivers de Cancer de Mama (TP53, ESR1, ERBB2, PGR, etc.)
  - Complementa con genes de alta varianza (HVGs) hasta alcanzar num_top_genes
  - Memoria optimizada y escalado robusto
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

from .pam50_biomarkers import ALL_BIOMARKER_GENES


def robust_scale(X: np.ndarray) -> np.ndarray:
    median = np.nanmedian(X, axis=0)
    q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (X - median) / iqr


class METABRICDataset(Dataset):
    """
    Dataset PyTorch multi-omico METABRIC v6.
    Combina genes PAM50/Drivers obligatorios + top HVGs + CNA + variables clinicas.
    """

    def __init__(
        self,
        metabric_dir: str = r"C:\Users\loapi\Downloads\brca_metabric\brca_metabric",
        num_top_genes: int = 500,
        is_training: bool = True
    ):
        self.metabric_dir = Path(metabric_dir)
        self.num_top_genes = num_top_genes
        self.is_training = is_training
        self.patient_ids, self.omic_features, self.survival_targets, self.feature_dim = \
            self._load_and_process_metabric()

    def _load_and_process_metabric(self) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        patient_file = self.metabric_dir / "data_clinical_patient.txt"
        sample_file  = self.metabric_dir / "data_clinical_sample.txt"
        mrna_file    = self.metabric_dir / "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
        cna_file     = self.metabric_dir / "data_cna.txt"

        if not patient_file.exists():
            print("[WARN] Archivos METABRIC no encontrados. Usando datos sinteticos.")
            n = 100
            feat_dim = self.num_top_genes * 2
            return ([f"MB-{i:04d}" for i in range(n)],
                    np.random.randn(n, feat_dim).astype(np.float32),
                    np.random.rand(n, 2).astype(np.float32),
                    feat_dim)

        # === 1. Clinical Patient ===
        print("[METABRIC] Cargando datos clinicos...")
        patient_df = pd.read_csv(patient_file, sep="\t", comment="#")
        cols_keep = ["PATIENT_ID", "RFS_MONTHS", "RFS_STATUS", "OS_MONTHS", "OS_STATUS",
                     "AGE_AT_DIAGNOSIS", "NPI", "CLAUDIN_SUBTYPE"]
        cols_keep = [c for c in cols_keep if c in patient_df.columns]
        patient_df = patient_df[cols_keep].copy()

        def parse_status(val):
            s = str(val)
            return 1.0 if ("1" in s or "Recurred" in s or "DECEASED" in s) else 0.0

        patient_df["rfs_event"] = patient_df["RFS_STATUS"].apply(parse_status)
        patient_df["rfs_time"]  = pd.to_numeric(patient_df["RFS_MONTHS"], errors="coerce")
        patient_df.dropna(subset=["rfs_time"], inplace=True)

        if sample_file.exists():
            sample_df = pd.read_csv(sample_file, sep="\t", comment="#")
            s_cols = ["PATIENT_ID", "ER_STATUS", "HER2_STATUS", "PR_STATUS", "GRADE"]
            s_cols = [c for c in s_cols if c in sample_df.columns]
            sample_df = sample_df[s_cols].drop_duplicates("PATIENT_ID")
            patient_df = pd.merge(patient_df, sample_df, on="PATIENT_ID", how="left")

        # === 2. mRNA Z-scores (PAM50 Biomarkers + Top HVG) ===
        print(f"[METABRIC] Cargando mRNA Z-scores (PAM50 + top {self.num_top_genes} HVG)...")
        mrna_df = pd.read_csv(mrna_file, sep="\t", comment="#")
        mrna_df.dropna(subset=["Hugo_Symbol"], inplace=True)
        mb_cols_mrna = [c for c in mrna_df.columns if c.startswith("MB-")]

        # Priorizar genes PAM50 y drivers que existan en mrna_df
        found_biomarkers = [g for g in ALL_BIOMARKER_GENES if g in mrna_df["Hugo_Symbol"].values]
        
        # Complementar con genes de alta varianza
        sample_rows = mrna_df[~mrna_df["Hugo_Symbol"].isin(found_biomarkers)].sample(
            n=min(2000, len(mrna_df) - len(found_biomarkers)), random_state=42
        )
        sample_vals = sample_rows[mb_cols_mrna].to_numpy(dtype=np.float32, na_value=0.0)
        sample_vars = np.var(sample_vals, axis=1)
        needed_hvg = max(0, self.num_top_genes - len(found_biomarkers))
        top_hvg_idx = np.argsort(sample_vars)[-needed_hvg:]
        hvg_genes = sample_rows.iloc[top_hvg_idx]["Hugo_Symbol"].values.tolist()

        final_gene_list = found_biomarkers + hvg_genes
        mrna_top_df = mrna_df[mrna_df["Hugo_Symbol"].isin(final_gene_list)].drop_duplicates("Hugo_Symbol")
        mrna_top = mrna_top_df[mb_cols_mrna].T
        mrna_top.index.name = "PATIENT_ID"
        mrna_top.reset_index(inplace=True)

        # === 3. CNA (PAM50 + HVG) ===
        cna_block = None
        if cna_file.exists():
            print(f"[METABRIC] Cargando CNA (PAM50 + top {self.num_top_genes} HVG)...")
            cna_df = pd.read_csv(cna_file, sep="\t", comment="#")
            cna_df.dropna(subset=["Hugo_Symbol"], inplace=True)
            mb_cols_cna = [c for c in cna_df.columns if c.startswith("MB-")]

            cna_biomarkers = [g for g in ALL_BIOMARKER_GENES if g in cna_df["Hugo_Symbol"].values]
            cna_sample_rows = cna_df[~cna_df["Hugo_Symbol"].isin(cna_biomarkers)].sample(
                n=min(2000, len(cna_df) - len(cna_biomarkers)), random_state=42
            )
            cna_sample_vals = cna_sample_rows[mb_cols_cna].to_numpy(dtype=np.float32, na_value=0.0)
            cna_sample_vars = np.var(cna_sample_vals, axis=1)
            cna_needed_hvg = max(0, self.num_top_genes - len(cna_biomarkers))
            cna_top_idx = np.argsort(cna_sample_vars)[-cna_needed_hvg:]
            cna_hvg_genes = cna_sample_rows.iloc[cna_top_idx]["Hugo_Symbol"].values.tolist()

            cna_final_genes = cna_biomarkers + cna_hvg_genes
            cna_top_df = cna_df[cna_df["Hugo_Symbol"].isin(cna_final_genes)].drop_duplicates("Hugo_Symbol")
            cna_top = cna_top_df[mb_cols_cna].T
            cna_top.index.name = "PATIENT_ID"
            cna_top.reset_index(inplace=True)
            cna_block = cna_top

        # === 4. Merge ===
        merged = pd.merge(patient_df, mrna_top, on="PATIENT_ID", how="inner")
        if cna_block is not None:
            merged = pd.merge(merged, cna_block, on="PATIENT_ID", how="left",
                              suffixes=("_mrna", "_cna"))

        patient_ids = merged["PATIENT_ID"].tolist()
        survival_targets = merged[["rfs_time", "rfs_event"]].values.astype(np.float32)

        clinical_feats = []
        for col in ["AGE_AT_DIAGNOSIS", "NPI"]:
            if col in merged.columns:
                v = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).values
                clinical_feats.append(v.reshape(-1, 1))

        if "CLAUDIN_SUBTYPE" in merged.columns:
            subtypes = ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal"]
            for s in subtypes:
                clinical_feats.append((merged["CLAUDIN_SUBTYPE"] == s).values.astype(np.float32).reshape(-1, 1))
        for col in ["ER_STATUS", "HER2_STATUS", "PR_STATUS"]:
            if col in merged.columns:
                clinical_feats.append((merged[col] == "Positive").values.astype(np.float32).reshape(-1, 1))

        skip_cols = set(["PATIENT_ID", "RFS_MONTHS", "RFS_STATUS", "OS_MONTHS", "OS_STATUS",
                         "rfs_event", "rfs_time", "AGE_AT_DIAGNOSIS", "NPI",
                         "CLAUDIN_SUBTYPE", "ER_STATUS", "HER2_STATUS", "PR_STATUS", "GRADE"])
        gene_cols = [c for c in merged.columns if c not in skip_cols]
        omic_matrix = merged[gene_cols].to_numpy(dtype=np.float32, na_value=0.0)
        omic_matrix = robust_scale(omic_matrix)

        if clinical_feats:
            clinical_matrix = np.hstack(clinical_feats).astype(np.float32)
            np.nan_to_num(clinical_matrix, nan=0.0, copy=False)
            omic_matrix = np.hstack([omic_matrix, clinical_matrix])

        feat_dim = omic_matrix.shape[1]
        print(f"[METABRIC v6] Listo: {len(patient_ids)} pacientes | "
              f"Feature vector: {feat_dim} dims (PAM50 + Drivers + HVG + CNA + Clinica)")
        return patient_ids, omic_matrix.astype(np.float32), survival_targets, feat_dim

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_ids[idx],
            "omic_tensor": torch.from_numpy(self.omic_features[idx]),
            "rfs_time":    torch.tensor(self.survival_targets[idx, 0], dtype=torch.float32),
            "rfs_event":   torch.tensor(self.survival_targets[idx, 1], dtype=torch.float32)
        }


def get_metabric_dataloader(
    metabric_dir: str = r"C:\Users\loapi\Downloads\brca_metabric\brca_metabric",
    batch_size: int = 32,
    shuffle: bool = True,
    num_top_genes: int = 500
) -> Tuple[DataLoader, int]:
    dataset = METABRICDataset(metabric_dir=metabric_dir, num_top_genes=num_top_genes, is_training=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset.feature_dim
