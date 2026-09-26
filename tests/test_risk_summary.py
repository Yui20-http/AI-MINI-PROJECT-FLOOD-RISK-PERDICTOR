import unittest

from app import build_risk_summary


class RiskSummaryTests(unittest.TestCase):
    def test_build_risk_summary_highlights_risk_drivers(self):
        summary = build_risk_summary(
            {
                "rainfall": "120",
                "humidity": "88",
                "river_discharge": "2400",
                "water_level": "3.4",
                "elevation": "12",
                "population_density": "6200",
                "infrastructure": "1",
                "historical_floods": "1",
            }
        )

        self.assertGreaterEqual(summary["pressure_score"], 60)
        self.assertIn(summary["risk_level"], {"Moderate", "High"})
        self.assertGreaterEqual(len(summary["actions"]), 3)
        self.assertTrue(any("rainfall" in driver["name"].lower() for driver in summary["drivers"]))


if __name__ == "__main__":
    unittest.main()
