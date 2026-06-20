# features/wake_healer.py
import numpy as np
import networkx as nx
from typing import List, Dict

class EmergencyWakeHealer:
    def __init__(self, traffic_graph=None):
        if traffic_graph is None:
            self.network = nx.grid_2d_graph(5, 5)
            mapping = {node: i for i, node in enumerate(self.network.nodes())}
            self.network = nx.relabel_nodes(self.network, mapping)
            for u, v in self.network.edges():
                self.network.edges[u, v]['queue_length'] = 0
                self.network.edges[u, v]['capacity'] = 100
        else:
            self.network = traffic_graph

    def detect_wake(self, ambulance_route: List[int]) -> List[int]:
        """
        Identifies intersections (nodes) affected by the emergency vehicle's wake.
        Ambulance splits traffic, leaving a tail of stopped/disrupted vehicles.
        """
        wake_nodes = []
        # Any intersection on the route gets affected.
        # Nodes adjacent to the route also experience spillback.
        for node in ambulance_route:
            if node in self.network:
                wake_nodes.append(node)
                # Add immediate neighbors experiencing spillback
                for neighbor in self.network.neighbors(node):
                    if neighbor not in wake_nodes and np.random.random() < 0.5:
                        wake_nodes.append(neighbor)
        return list(set(wake_nodes))

    def reconstruct_flow(self, wake_nodes: List[int]) -> Dict[int, dict]:
        """
        Recommend traffic signal overrides (holding greens) to clear the ambulance wake.
        Target: recover normal flow in 90 seconds.
        """
        overrides = {}
        for node in wake_nodes:
            # Recommend custom green phase extension in seconds based on connectivity
            degree = self.network.degree(node) if node in self.network else 2
            green_extension = int(min(30, max(10, degree * 6)))
            overrides[node] = {
                "signal_id": f"SIG_{node}",
                "green_extension_sec": green_extension,
                "action": "Hold Green Phase",
                "reason": "Clear emergency vehicle split-platoon queue spillback."
            }
        return overrides

    def simulate_recovery(self, wake_nodes: List[int], strategy_active: bool = True) -> dict:
        """
        Simulates queue lengths over 5 time steps (each step represents 20 seconds).
        Total duration: 100 seconds. Shows recovery rate.
        """
        initial_queues = {node: int(np.random.randint(15, 30)) for node in wake_nodes}
        current_queues = initial_queues.copy()
        
        timeline = []
        for step in range(5):
            # Queue reduction rate: without strategy, clearance is slow (3 cars/step)
            # With signal override strategy, clearance is fast (8 cars/step)
            clearance_rate = 8 if strategy_active else 3
            
            for node in wake_nodes:
                current_queues[node] = int(max(0, current_queues[node] - clearance_rate + np.random.randint(0, 2)))
                
            avg_queue = float(np.mean(list(current_queues.values())))
            timeline.append({
                "seconds_elapsed": (step + 1) * 20,
                "avg_queue_len": round(avg_queue, 1),
                "cleared_pct": round((1.0 - (avg_queue / np.mean(list(initial_queues.values())))) * 100, 1)
            })
            
        return {
            "initial_avg_queue": round(float(np.mean(list(initial_queues.values()))), 1),
            "final_avg_queue": round(float(np.mean(list(current_queues.values()))), 1),
            "timeline": timeline,
            "success": bool(np.mean(list(current_queues.values())) < 3.0)
        }
