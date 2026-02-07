import unittest
from unittest.mock import patch
from src.tools.greeter_tools import lookup_customer, verify_answer

class TestGreeterTools(unittest.TestCase):

    def setUp(self):
        self.mock_customers = {
            "customers": [
                {
                    "name": "Test User",
                    "phone": "+123456789",
                    "iban": "DE123456789",
                    "secret": "What is the secret?",
                    "answer": "The Answer"
                },
                {
                    "name": "Another User",
                    "phone": "+987654321",
                    "iban": "DE987654321",
                    "secret": "Another secret?",
                    "answer": "Another Answer"
                }
            ]
        }

    @patch('src.tools.greeter_tools.load_customers_data')
    def test_lookup_customer_insufficient_details(self, mock_load_data):
        mock_load_data.return_value = self.mock_customers
        
        # 0 details
        result = lookup_customer.invoke({})
        self.assertIn("Error: You must provide at least two details", result)
        
        # 1 detail
        result = lookup_customer.invoke({"name": "Test User"})
        self.assertIn("Error: You must provide at least two details", result)

    @patch('src.tools.greeter_tools.load_customers_data')
    def test_lookup_customer_success(self, mock_load_data):
        mock_load_data.return_value = self.mock_customers
        
        # Name + Phone
        result = lookup_customer.invoke({"name": "Test User", "phone": "+123456789"})
        self.assertIn("Customer found", result)
        self.assertIn("What is the secret?", result)
        
        # Name + IBAN
        result = lookup_customer.invoke({"name": "Test User", "iban": "DE123456789"})
        self.assertIn("Customer found", result)
        
        # Phone + IBAN
        result = lookup_customer.invoke({"phone": "+123456789", "iban": "DE123456789"})
        self.assertIn("Customer found", result)

    @patch('src.tools.greeter_tools.load_customers_data')
    def test_lookup_customer_mismatch(self, mock_load_data):
        mock_load_data.return_value = self.mock_customers
        
        # Matching name but wrong phone
        result = lookup_customer.invoke({"name": "Test User", "phone": "+987654321"})
        self.assertIn("Customer not found", result)
        
        # Non-existent user
        result = lookup_customer.invoke({"name": "Non Existent", "phone": "+000000000"})
        self.assertIn("Customer not found", result)

    @patch('src.tools.greeter_tools.load_customers_data')
    def test_verify_answer_success(self, mock_load_data):
        mock_load_data.return_value = self.mock_customers
        
        # Correct answer
        result = verify_answer.invoke({"name": "Test User", "phone": "+123456789", "answer": "The Answer"})
        self.assertIn("VERIFIED", result)
        
        # Case insensitive
        result = verify_answer.invoke({"name": "Test User", "phone": "+123456789", "answer": "the answer"})
        self.assertIn("VERIFIED", result)

    @patch('src.tools.greeter_tools.load_customers_data')
    def test_verify_answer_incorrect(self, mock_load_data):
        mock_load_data.return_value = self.mock_customers
        
        result = verify_answer.invoke({"name": "Test User", "phone": "+123456789", "answer": "Wrong Answer"})
        self.assertIn("Incorrect answer", result)

    @patch('src.tools.greeter_tools.load_customers_data')
    def test_verify_answer_customer_not_found(self, mock_load_data):
        mock_load_data.return_value = self.mock_customers
        
        result = verify_answer.invoke({"name": "Non Existent", "phone": "+123456789", "answer": "The Answer"})
        self.assertIn("Customer not found", result)

if __name__ == '__main__':
    unittest.main()
