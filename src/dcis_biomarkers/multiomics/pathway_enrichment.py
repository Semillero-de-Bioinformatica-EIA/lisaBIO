import pandas as pd
from typing import List, Dict, Any

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

def run_enrichment(gene_list: List[str], gene_sets: str = 'KEGG_2021_Human') -> pd.DataFrame:
    """
    Ejecuta Pathway Enrichment Analysis (ORA) para una lista de genes.
    """
    if not HAS_GSEAPY:
        raise ImportError("La librería gseapy no está instalada. Ejecute: pip install gseapy")
        
    if not gene_list:
        return pd.DataFrame()

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism='Human',
            outdir=None
        )
        if enr.results is not None:
            return enr.results
        else:
            return pd.DataFrame()
    except Exception as e:
        # En lugar de fabricar resultados (0.001 p-values falsos), fallamos explícitamente
        # o devolvemos un DataFrame vacío con error loggeado
        import logging
        logging.getLogger(__name__).error(f"Fallo en GSEApy enrichr: {e}")
        raise RuntimeError(f"Error en el análisis de pathway enrichment: {e}")
