# features/prophecy_breaker.py
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Driver:
    id: int
    origin: int
    destination: int
    current_route: List[int]

@dataclass
class RoadState:
    road_id: int
    capacity: int
    current_flow: int
    travel_time: float

class ProphecyBreakerRouter:
    def __init__(self, road_network: nx.Graph = None):
        if road_network is None:
            # Generate a synthetic 50-node grid graph representing MG Road, Koramangala area
            self.network = nx.grid_2d_graph(5, 10)
            # Relabel nodes to integers
            mapping = {node: i for i, node in enumerate(self.network.nodes())}
            self.network = nx.relabel_nodes(self.network, mapping)
            # Add capacities and baseline free flow times
            for u, v in self.network.edges():
                self.network.edges[u, v]['capacity'] = np.random.randint(40, 100)
                self.network.edges[u, v]['time'] = np.random.uniform(0.5, 2.0)
        else:
            self.network = road_network
            
        self.equilibrium_tolerance = 0.05
        self.max_iterations = 5
        self.collapse_penalty = 0.3

    def compute_equilibrium_routes(self, drivers: List[Driver]) -> Dict[int, List[int]]:
        """
        Compute routes that form Nash equilibrium.
        No driver should have incentive to deviate after everyone follows recommendations.
        """
        # Step 1: Initial route assignment (shortest path)
        routes = {}
        for d in drivers:
            try:
                routes[d.id] = nx.shortest_path(self.network, d.origin, d.destination, weight='time')
            except nx.NetworkXNoPath:
                routes[d.id] = [d.origin, d.destination]

        # Step 2: Iterate toward equilibrium
        for iteration in range(self.max_iterations):
            predicted_congestion = self._simulate_adoption(routes, drivers)
            if self._is_equilibrium_stable(routes, predicted_congestion, drivers):
                return routes
            routes = self._rebalance_routes(routes, predicted_congestion, drivers)
        return routes

    def simulate_recommendation_impact(self, routes: Dict[int, List[int]], drivers: List[Driver]) -> Dict[Tuple[int, int], float]:
        """
        Simulate travel times after recommendation is adopted.
        """
        return self._simulate_adoption(routes, drivers)

    def _simulate_adoption(self, routes, drivers):
        """Simulate what happens when all drivers follow the routes."""
        congestion = {edge: 0 for edge in self.network.edges()}
        for driver in drivers:
            route = routes.get(driver.id)
            if not route or len(route) < 2:
                continue
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                edge = (u, v) if (u, v) in congestion else (v, u)
                if edge in congestion:
                    congestion[edge] += 1

        predicted_travel_time = {}
        for edge in self.network.edges():
            flow = congestion.get(edge, 0)
            capacity = self.network.edges[edge].get('capacity', 100)
            free_flow_time = self.network.edges[edge].get('time', 1.0)
            ratio = flow / capacity
            predicted_travel_time[edge] = free_flow_time * (1 + 0.15 * (ratio ** 4))
        return predicted_travel_time

    def _is_equilibrium_stable(self, routes, congestion, drivers):
        """Check if any driver can improve by switching routes."""
        for driver in drivers:
            current_route = routes.get(driver.id)
            if not current_route or len(current_route) < 2:
                continue
            
            current_time = 0
            for i in range(len(current_route) - 1):
                u, v = current_route[i], current_route[i+1]
                edge = (u, v) if (u, v) in congestion else (v, u)
                current_time += congestion.get(edge, 1.0)

            for u, v in self.network.edges():
                edge = (u, v) if (u, v) in congestion else (v, u)
                self.network.edges[u, v]['temp_weight'] = congestion.get(edge, 1.0)

            try:
                alt_route = nx.shortest_path(self.network, driver.origin, driver.destination, weight='temp_weight')
                alt_time = sum(self.network.edges[alt_route[i], alt_route[i+1]]['temp_weight'] for i in range(len(alt_route) - 1))
            except Exception:
                continue

            if alt_time < current_time * (1 - self.equilibrium_tolerance):
                return False
        return True

    def _rebalance_routes(self, routes, congestion, drivers):
        """Reassign drivers from overloaded routes to alternatives using fast randomized Dijkstra."""
        sorted_roads = sorted(congestion.items(), key=lambda x: x[1], reverse=True)
        overloaded = [r[0] for r in sorted_roads[:3]]

        for driver in drivers:
            route = routes.get(driver.id)
            if not route or len(route) < 2:
                continue
            
            uses_overloaded = False
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                edge = (u, v) if (u, v) in congestion else (v, u)
                if edge in overloaded:
                    uses_overloaded = True
                    break
                    
            if uses_overloaded and np.random.random() < 0.2:
                try:
                    # Perturb weights slightly to find random fast alternative
                    for u, v in self.network.edges():
                        edge = (u, v) if (u, v) in congestion else (v, u)
                        self.network.edges[u, v]['temp_perturbed'] = congestion.get(edge, 1.0) * np.random.uniform(0.9, 1.1)
                        
                    alt_route = nx.shortest_path(self.network, driver.origin, driver.destination, weight='temp_perturbed')
                    routes[driver.id] = alt_route
                except Exception:
                    pass
        return routes
