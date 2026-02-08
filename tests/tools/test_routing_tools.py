import unittest
from src.tools.specialist_tools import route_to_expert
from src.tools.bouncer_tools import handoff_to_specialist


class TestRoutingTools(unittest.TestCase):

    def test_route_to_expert_yacht_insurance(self):
        result = route_to_expert.invoke({"category": "yacht_insurance"})
        self.assertIn("Yacht & Marine Insurance", result)
        self.assertIn("+1999888001", result)

    def test_route_to_expert_wealth_management(self):
        result = route_to_expert.invoke({"category": "wealth_management"})
        self.assertIn("Wealth Management", result)
        self.assertIn("+1999888002", result)

    def test_route_to_expert_real_estate(self):
        result = route_to_expert.invoke({"category": "real_estate"})
        self.assertIn("Real Estate", result)
        self.assertIn("+1999888003", result)

    def test_route_to_expert_general_premium(self):
        result = route_to_expert.invoke({"category": "general_premium"})
        self.assertIn("Premium General Support", result)
        self.assertIn("+1999888004", result)

    def test_route_to_expert_invalid_category(self):
        result = route_to_expert.invoke({"category": "invalid_category"})
        self.assertIn("Error", result)
        self.assertIn("invalid_category", result)

    def test_handoff_to_specialist(self):
        result = handoff_to_specialist.invoke({})
        self.assertIn("Handing off to Specialist", result)


if __name__ == '__main__':
    unittest.main()
