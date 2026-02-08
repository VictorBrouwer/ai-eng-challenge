import unittest
from unittest.mock import patch, MagicMock
from src.tools.bouncer_tools import check_account_status

class TestAccountTools(unittest.TestCase):

    def setUp(self):
        self.mock_data = {
            "customers": [],
            "accounts": [
                {"iban": "PREMIUM_IBAN", "premium": True},
                {"iban": "REGULAR_IBAN", "premium": False}
            ]
        }

    @patch('src.tools.bouncer_tools.load_customers_data')
    def test_check_account_status_premium(self, mock_load_data):
        mock_load_data.return_value = self.mock_data
        result = check_account_status.invoke("PREMIUM_IBAN")
        self.assertEqual(result, "Premium")

    @patch('src.tools.bouncer_tools.load_customers_data')
    def test_check_account_status_regular(self, mock_load_data):
        mock_load_data.return_value = self.mock_data
        result = check_account_status.invoke("REGULAR_IBAN")
        self.assertEqual(result, "Regular")

    @patch('src.tools.bouncer_tools.load_customers_data')
    def test_check_account_status_non_client(self, mock_load_data):
        mock_load_data.return_value = self.mock_data
        result = check_account_status.invoke("NON_EXISTENT_IBAN")
        self.assertEqual(result, "Non-Client")

    @patch('src.tools.bouncer_tools.load_customers_data')
    def test_check_account_status_whitespace(self, mock_load_data):
        mock_load_data.return_value = self.mock_data
        result = check_account_status.invoke("  PREMIUM_IBAN  ")
        self.assertEqual(result, "Premium")

if __name__ == '__main__':
    unittest.main()
