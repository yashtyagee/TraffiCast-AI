# features/route_diversity.py
import networkx as nx
import numpy as np
from typing import List, Dict

class RouteDiversityDefender:
    def __init__(self, road_network=None):
        if road_network is None:
            # Generate synthetic grid network
            self.network = nx.grid_2d_graph(6, 6)
            mapping = {node: i for i, node in enumerate(self.network.nodes())}
            self.network = nx.relabel_nodes(self.network, mapping)
            for u, v in self.network.edges():
                self.network.edges[u, v]['time'] = np.random.uniform(1.0, 3.0)
        else:
            self.network = road_network
            
        self.diversity_weight = 0.3
        self.max_acceptable_detour = 1.25  # up to 25% detour allowed for robustness

    def compute_diverse_routes(self, drivers: list, K: int = 5) -> Dict[int, List[int]]:
        """
        Maximize route entropy while keeping travel time detours within acceptable bounds.
        """
        route_assignment = {}
        route_usage = {}

        for d in drivers:
            paths = []
            try:
                # Find optimal shortest path
                opt_path = nx.shortest_path(self.network, d.origin, d.destination, weight='time')
                paths.append(opt_path)
                
                # Generate up to K alternatives by blocking edges on the optimal path
                for idx in range(len(opt_path) - 1):
                    u, v = opt_path[idx], opt_path[idx+1]
                    orig_weight = self.network.edges[u, v]['time']
                    self.network.edges[u, v]['time'] = orig_weight * 10.0
                    try:
                        alt_path = nx.shortest_path(self.network, d.origin, d.destination, weight='time')
                        if alt_path not in paths:
                            paths.append(alt_path)
                    except Exception:
                        pass
                    self.network.edges[u, v]['time'] = orig_weight
                    if len(paths) >= K:
                        break
            except Exception:
                paths = [[d.origin, d.destination]]
                
            optimal_time = sum(self.network.edges[paths[0][i], paths[0][i+1]]['time'] for i in range(len(paths[0]) - 1))
            
            # Keep only paths within acceptable detour threshold
            acceptable_paths = []
            for p in paths:
                p_time = sum(self.network.edges[p[i], p[i+1]]['time'] for i in range(len(p) - 1))
                if p_time <= optimal_time * self.max_acceptable_detour:
                    acceptable_paths.append(p)
                    
            if not acceptable_paths:
                acceptable_paths = [paths[0]]
                
            # Greedy allocation: pick the route currently used the least
            least_used = min(acceptable_paths, key=lambda r: route_usage.get(tuple(r), 0))
            
            route_assignment[d.id] = least_used
            route_usage[tuple(least_used)] = route_usage.get(tuple(least_used), 0) + 1

        return route_assignment

    def compute_route_entropy(self, route_assignment: Dict[int, List[int]]) -> float:
        """Calculate Shannon entropy of the assigned routes."""
        if not route_assignment:
            return 0.0
        route_counts = {}
        for r in route_assignment.values():
            rt = tuple(r)
            route_counts[rt] = route_counts.get(rt, 0) + 1
            
        total = len(route_assignment)
        probs = [count / total for count in route_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return float(entropy)

    def assess_fragility(self, route_assignment: Dict[int, List[int]]) -> float:
        """
        Calculates a fragility score between 0.0 (highly robust) and 1.0 (highly fragile).
        Fragility decreases with higher route entropy.
        """
        if not route_assignment:
            return 0.0
        unique_routes = len(set(tuple(r) for r in route_assignment.values()))
        if unique_routes <= 1:
            return 1.0
            
        max_entropy = np.log2(unique_routes)
        actual_entropy = self.compute_route_entropy(route_assignment)
        
        # Fragility index
        fragility = 1.0 - (actual_entropy / max_entropy if max_entropy > 0 else 1.0)
        return float(np.clip(fragility, 0.0, 1.0))
