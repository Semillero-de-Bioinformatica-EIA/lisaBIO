import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.spatial import distance_matrix

def build_spatial_microenvironment_graph(
    coordinates: np.ndarray, 
    features: np.ndarray, 
    k_neighbors: int = 8
) -> Dict[str, Any]:
    """
    Construye un grafo espacial del microambiente tumoral representando las interacciones
    entre celulas/parches de CDIS, estroma y linfocitos infiltrantes de tumor (TILs).
    """
    num_nodes = coordinates.shape[0]
    dist_mat = distance_matrix(coordinates, coordinates)

    edge_index = []
    for i in range(num_nodes):
        # Obtener los k vecinos más cercanos excluyéndose a sí mismo
        nearest_indices = np.argsort(dist_mat[i])[1:k_neighbors + 1]
        for neighbor in nearest_indices:
            edge_index.append((i, neighbor))

    return {
        "x": features,                  # Atributos de nodos (vectores de características)
        "edge_index": np.array(edge_index).T, # Aristas (2, N_aristas)
        "pos": coordinates              # Coordenadas espaciales 2D en WSI
    }
