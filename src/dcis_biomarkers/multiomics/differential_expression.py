import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def run_differential_expression(df_expr: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Calcula la expresión diferencial simple.
    df_expr: DataFrame de genes (pacientes x genes)
    labels: Serie binaria (e.g., 0 = Indolent, 1 = Progressive) indexada por paciente.
    """
    if not HAS_STATSMODELS:
        raise ImportError("La librería statsmodels no está instalada. Ejecute: pip install statsmodels")

    common_idx = df_expr.index.intersection(labels.index)
    X = df_expr.loc[common_idx]
    y = labels.loc[common_idx]

    results = []
    
    # Asumiendo y contiene 0 y 1
    group_0 = y == 0
    group_1 = y == 1

    if sum(group_0) == 0 or sum(group_1) == 0:
        raise ValueError("No hay suficientes muestras en al menos uno de los grupos para DE.")

    for gene in X.columns:
        expr = X[gene].values
        
        # Simple fold change (log2 fold change si expr ya está en log scale)
        mean_0 = expr[group_0].mean()
        mean_1 = expr[group_1].mean()
        log2fc = mean_1 - mean_0
        
        # T-test asumiendo varianzas diferentes
        try:
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(expr[group_1], expr[group_0], equal_var=False)
        except Exception:
            p_val = 1.0

        results.append({
            "gene": gene,
            "log2fc": log2fc,
            "p_value": p_val
        })

    df_res = pd.DataFrame(results)
    
    # FDR Correction
    from statsmodels.stats.multitest import multipletests
    if not df_res.empty:
        _, p_adj, _, _ = multipletests(df_res["p_value"].fillna(1.0), alpha=0.05, method='fdr_bh')
        df_res["padj"] = p_adj
    else:
        df_res["padj"] = []

    return df_res.sort_values("p_value")
