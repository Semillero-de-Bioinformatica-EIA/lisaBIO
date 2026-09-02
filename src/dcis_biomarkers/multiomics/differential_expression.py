import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def run_differential_expression(
    expr_df: pd.DataFrame, 
    group_labels: pd.Series, 
    group_cases: str = "Invasive", 
    group_controls: str = "DCIS_Indolent"
) -> pd.DataFrame:
    """
    Ejecuta prueba de expresión diferencial (Wilcoxon rank-sum) entre muestras de CDIS progresivo/invasivo vs indolente.
    Retorna p-values, p-adjusted (FDR) y log2 Fold Change (log2FC).
    """
    cases_mask = group_labels == group_cases
    controls_mask = group_labels == group_controls

    cases_data = expr_df.loc[cases_mask]
    controls_data = expr_df.loc[controls_mask]

    results = []

    for gene in expr_df.columns:
        c_vals = cases_data[gene].values
        ctrl_vals = controls_data[gene].values

        mean_case = np.mean(c_vals)
        mean_ctrl = np.mean(ctrl_vals)
        log2fc = mean_case - mean_ctrl  # Asumiendo datos log-transformados

        try:
            stat, pval = stats.mannwhitneyu(c_vals, ctrl_vals, alternative='two-sided')
        except Exception:
            pval = 1.0

        results.append({
            'gene': gene,
            'log2FC': log2fc,
            'pval': pval,
            'mean_case': mean_case,
            'mean_control': mean_ctrl
        })

    res_df = pd.DataFrame(results)
    _, padj, _, _ = multipletests(res_df['pval'].values, method='fdr_bh')
    res_df['padj'] = padj
    return res_df.sort_values('padj')
