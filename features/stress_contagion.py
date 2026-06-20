# features/stress_contagion.py
import numpy as np
import networkx as nx
from typing import Dict

class StressContagionPredictor:
    def __init__(self, traffic_graph=None):
        if traffic_graph is None:
            # Generate synthetic traffic network
            self.graph = nx.erdos_renyi_graph(50, 0.1, seed=42)
            for u, v in self.graph.edges():
                self.graph.edges[u, v]['stress_multiplier'] = np.random.uniform(1.0, 2.0)
        else:
            self.graph = traffic_graph
            
        self.beta = 0.4   # stress transmission rate
        self.gamma = 0.1  # recovery rate

    def predict_stress_spread(self, current_states: Dict[int, str], time_horizon_min: int = 15) -> Dict[int, str]:
        """
        current_states: Dict[driver_id, 'S'|'I'|'R']
        Returns: predicted states after time_horizon_min minutes
        """
        states = current_states.copy()
        
        # Build simple adjacency for drivers based on synthetic graph nodes
        driver_nodes = list(states.keys())
        
        for minute in range(time_horizon_min):
            new_states = states.copy()

            for driver_id, state in states.items():
                if state == 'S':
                    # Count infected neighbors
                    # Since it's a grid/network, we assume drivers close in ID or connected in graph share stress
                    infected_neighbors = 0
                    for other_id in driver_nodes:
                        if other_id != driver_id and states[other_id] == 'I':
                            # Simple distance probability: close IDs or graph connection
                            if abs(driver_id - other_id) < 3 or self.graph.has_edge(driver_id % 50, other_id % 50):
                                infected_neighbors += 1
                                
                    infection_prob = 1 - np.exp(-self.beta * infected_neighbors * 0.2)
                    if np.random.random() < infection_prob:
                        new_states[driver_id] = 'I'

                elif state == 'I':
                    # Recovery back to calm
                    if np.random.random() < self.gamma:
                        new_states[driver_id] = 'R'

            states = new_states
        return states

    def predict_outbreak(self, current_states: Dict[int, str]) -> bool:
        """Returns True if stress outbreak is predicted in the next 15 min."""
        future_states = self.predict_stress_spread(current_states, 15)
        infected_count = sum(1 for s in future_states.values() if s == 'I')
        return infected_count > len(current_states) * 0.15  # 15% threshold

    def suggest_intervention(self, current_states: Dict[int, str]) -> dict:
        """Recommend calming intervention in highest-risk zone."""
        future_states = self.predict_stress_spread(current_states, 15)
        
        # Group drivers by synthetic zones (e.g. driver_id % 4)
        zone_infections = {0: 0, 1: 0, 2: 0, 3: 0}
        for driver_id, state in future_states.items():
            if state == 'I':
                zone_infections[driver_id % 4] += 1
                
        highest_risk_zone = max(zone_infections, key=zone_infections.get)
        
        interventions = {
            0: {"name": "Dynamic Speed Limit Reduction", "desc": "Reduce speed limit to 40 km/h in Sector Alpha to minimize sudden lane changes.", "calming_impact": "High (-35% stress)"},
            1: {"name": "Green Wave Signal Harmonization", "desc": "Hold green lights longer on Corridor Beta to reduce stop-and-go irritation.", "calming_impact": "Critical (-50% stress)"},
            2: {"name": "Preemptive Diversion Nudge", "desc": "Redirect incoming vehicles to alternate routes to decrease density in Sector Gamma.", "calming_impact": "Medium (-20% stress)"},
            3: {"name": "Digital Signage Calm Notice", "desc": "Display 'Relax, traffic ahead' on VMS boards in Sector Delta to align driver expectations.", "calming_impact": "Low (-10% stress)"}
        }
        
        return {
            "zone": f"Sector {chr(65 + highest_risk_zone)}",
            "active_infections": zone_infections[highest_risk_zone],
            "recommended_action": interventions[highest_risk_zone]["name"],
            "action_details": interventions[highest_risk_zone]["desc"],
            "calming_impact": interventions[highest_risk_zone]["calming_impact"]
        }
