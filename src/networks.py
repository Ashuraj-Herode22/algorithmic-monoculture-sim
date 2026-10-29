"""
Network Topologies Module
Generates Watts-Strogatz (Small-World) and Barabási-Albert (Scale-Free) graphs
and constructs row-stochastic DeGroot influence matrices.
"""

from typing import Dict, Any, Tuple
import networkx as nx
import numpy as np


def generate_watts_strogatz_graph(
    n: int, k: int = 6, p: float = 0.1, seed: int = None
) -> nx.Graph:
    """
    Generates a Watts-Strogatz small-world graph.

    Args:
        n: Number of nodes.
        k: Each node is connected to k nearest neighbors in ring topology.
        p: Rewiring probability (p=0: regular lattice, p=1: random graph).
        seed: Random seed for reproducibility.

    Returns:
        Connected or largest connected component of nx.Graph.
    """
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    # Ensure network is connected for proper consensus dynamics
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def generate_barabasi_albert_graph(
    n: int, m: int = 3, seed: int = None
) -> nx.Graph:
    """
    Generates a Barabási-Albert scale-free graph using preferential attachment.

    Args:
        n: Number of nodes.
        m: Number of edges to attach from a new node to existing nodes.
        seed: Random seed for reproducibility.

    Returns:
        nx.Graph (guaranteed connected for m >= 1).
    """
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def compute_network_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Computes key network diagnostic metrics:
        - Node count and Edge count
        - Average Clustering Coefficient (C)
        - Average Shortest Path Length (L)
        - Degree distribution statistics (mean, max, std)
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    clustering_coeff = nx.average_clustering(G)

    # Average shortest path length (on largest component if disconnected)
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        avg_path_length = nx.average_shortest_path_length(subG)

    degrees = [d for _, d in G.degree()]
    mean_degree = float(np.mean(degrees))
    max_degree = int(np.max(degrees))
    degree_std = float(np.std(degrees))

    return {
        "num_nodes": n,
        "num_edges": m,
        "avg_clustering_coeff": float(clustering_coeff),
        "avg_path_length": float(avg_path_length),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "degree_std": degree_std,
    }


def get_degroot_trust_matrix(G: nx.Graph, self_weight: float = 0.5) -> np.ndarray:
    """
    Constructs a row-stochastic DeGroot trust matrix W from graph G.

    Row i specifies the weights agent i assigns to neighbors and themselves:
        W_ii = self_weight
        W_ij = (1.0 - self_weight) / degree(i)  for j in neighbors(i)

    Each row sums to exactly 1.0 (stochastic matrix).
    """
    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    W = np.zeros((n, n), dtype=np.float64)

    for node in nodes:
        i = node_to_idx[node]
        neighbors = list(G.neighbors(node))
        k = len(neighbors)

        if k == 0:
            W[i, i] = 1.0
        else:
            W[i, i] = self_weight
            neighbor_weight = (1.0 - self_weight) / k
            for neighbor in neighbors:
                j = node_to_idx[neighbor]
                W[i, j] = neighbor_weight

    return W

