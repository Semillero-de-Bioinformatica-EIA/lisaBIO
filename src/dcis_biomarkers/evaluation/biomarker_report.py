import pandas as pd
import numpy as np

def generate_biomarker_report(de_results: pd.DataFrame, pathway_results: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida resultados de expresión diferencial y enriquecimiento de vías 
    en un reporte único de biomarcadores validados estadísticamente.
    """
    # Filtrar DE por p-adj (FDR)
    sig_de = de_results[de_results["padj"] < 0.05].copy()
    
    # Añadir columna de dirección
    if not sig_de.empty:
        sig_de["direction"] = np.where(sig_de["log2fc"] > 0, "Up-regulated", "Down-regulated")
    else:
        sig_de["direction"] = []

    # En la práctica, se cruzan los genes significativos con las vías enriquecidas.
    # Esta es una versión simplificada.
    
    report_rows = []
    for _, row in sig_de.iterrows():
        gene = row["gene"]
        # Encontrar en qué vías significativas aparece este gen
        pathways = []
        if not pathway_results.empty and "Term" in pathway_results.columns and "Genes" in pathway_results.columns:
            sig_pathways = pathway_results[pathway_results["Adjusted P-value"] < 0.05]
            for _, pw_row in sig_pathways.iterrows():
                if isinstance(pw_row["Genes"], str) and gene in pw_row["Genes"].split(";"):
                    pathways.append(pw_row["Term"])
                    
        report_rows.append({
            "Biomarker_ID": gene,
            "Modality": "Transcriptomics", # Se asume en este ejemplo
            "Effect_Size_log2FC": row["log2fc"],
            "P_Value": row["p_value"],
            "FDR_P_Value": row["padj"],
            "Direction": row["direction"],
            "Associated_Pathways": "; ".join(pathways) if pathways else "None"
        })
        
    return pd.DataFrame(report_rows)
