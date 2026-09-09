import numpy as np

def build_spatial_microenvironment_graph(coords: np.ndarray, features: np.ndarray, k_neighbors: int = 5, distance_threshold_um: float = 100.0) -> dict:
    """
    Construye un grafo espacial a partir de coordenadas y características.
    Usa distancias reales en micrómetros y k-NN.
    
    Args:
        coords: (N, 2) en micrones.
        features: (N, D).
        k_neighbors: número de vecinos.
        distance_threshold_um: distancia máxima para considerar una arista.
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(coords) == 0:
        return {"x": np.empty((0, features.shape[1])), "edge_index": np.empty((2, 0))}
        
    k = min(k_neighbors, len(coords) - 1)
    if k <= 0:
        return {"x": features, "edge_index": np.empty((2, 0))}
        
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    edges = []
    for i in range(len(coords)):
        for j_idx in range(1, k+1): # Ignorar el vecino 0 que es él mismo
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            if dist <= distance_threshold_um:
                edges.append((i, j))
                
    if not edges:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T
        
    return {
        "x": features,
        "edge_index": edge_index
    }
