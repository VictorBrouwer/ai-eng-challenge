import unittest
from unittest.mock import patch, MagicMock
from src.agents.greeter import greeter_node
from langchain_core.messages import HumanMessage, AIMessage

class TestGreeterAgent(unittest.TestCase):
    
    @patch('src.agents.greeter.ChatOpenAI')
    def test_greeter_node(self, mock_chat):
        # Setup mock
        mock_model_instance = MagicMock()
        mock_chat.return_value = mock_model_instance
        mock_model_with_tools = MagicMock()
        mock_model_instance.bind_tools.return_value = mock_model_with_tools
        
        expected_response = AIMessage(content="Hello")
        mock_model_with_tools.invoke.return_value = expected_response
        
        # Run node
        state = {"messages": [HumanMessage(content="Hi")]}
        result = greeter_node(state)
        
        # Verify
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], expected_response)
        mock_model_with_tools.invoke.assert_called_once()

if __name__ == '__main__':
    unittest.main()
