# features/induced_demand_oracle.py
import numpy as np

class InducedDemandOracle:
    def __init__(self):
        self.demand_elasticity = 0.4  # default urban elasticity coefficient
        self.short_term_horizon_days = 7
        self.long_term_horizon_months = 6

    def assess_intervention(self, intervention_name: str, travel_time_reduction_pct: float) -> dict:
        """
        Assess short-term and long-term impact of adding road capacity/optimizing routes.
        """
        # Short-term: positive benefit
        short_term_benefit = travel_time_reduction_pct

        # Long-term: induced demand erodes benefit (elasticity factor)
        latent_demand_pct = travel_time_reduction_pct * self.demand_elasticity
        long_term_congestion_increase = latent_demand_pct * 0.8  # 80% materializes over time

        net_long_term_effect = short_term_benefit - long_term_congestion_increase

        explanation = (
            f"Adding capacity reduces transit times by {travel_time_reduction_pct:.0f}% initially. "
            f"However, this drop in travel time attracts {latent_demand_pct:.1f}% new vehicle trips (latent demand) "
            f"which will fill the extra space, eroding {long_term_congestion_increase:.1f}% of the initial relief within 6 months."
        )

        return {
            'intervention': intervention_name,
            'short_term_benefit_pct': round(short_term_benefit, 1),
            'induced_demand_pct': round(latent_demand_pct, 1),
            'long_term_net_effect_pct': round(net_long_term_effect, 1),
            'recommendation': 'PROCEED' if net_long_term_effect > 8.0 else 'AVOID & OPTIMIZE ALTERNATIVES',
            'explanation': explanation
        }

    def suggest_demand_neutral_alternative(self, intervention_name: str) -> list[dict]:
        """Suggest alternatives that achieve same relief without induced demand."""
        return [
            {
                'name': 'Off-Peak Nudging & Variable Tolling',
                'description': 'Implement peak congestion pricing to shift 8% of commute trips to off-peak slots.',
                'induced_demand_pct': 0.0,
                'congestion_relief_pct': 10.0
            },
            {
                'name': 'Corridor Micro-Transit Integration',
                'description': 'Deploy 20 high-frequency electric shuttle lanes on the affected corridor.',
                'induced_demand_pct': -2.5, # net transit reduction
                'congestion_relief_pct': 15.0
            },
            {
                'name': 'High-Occupancy Vehicle (HOV) Dedicated Lanes',
                'description': 'Restructure existing lanes to HOV-3 only, increasing passenger throughput without road expansion.',
                'induced_demand_pct': 1.0,
                'congestion_relief_pct': 12.0
            }
        ]
