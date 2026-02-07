import unittest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph.routing import greeter_router
from src.graph.state import State

class TestRouting(unittest.TestCase):
    
    def test_continue_greeter_no_messages(self):
        state = {"messages": []}
        result = greeter_router(state)
        self.assertEqual(result, "continue_greeter")

    def test_call_tool(self):
        # Correctly mocking a tool call message
        msg = AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])
        state = {"messages": [msg]}
        result = greeter_router(state)
        self.assertEqual(result, "call_tool")

    def test_go_to_bouncer_verified(self):
        msg = ToolMessage(content='{"status": "VERIFIED"}', tool_call_id="1", name="verify_answer")
        state = {"messages": [msg]}
        result = greeter_router(state)
        self.assertEqual(result, "go_to_bouncer")
        
    def test_continue_greeter_not_verified(self):
        msg = ToolMessage(content="Incorrect answer", tool_call_id="1", name="verify_answer")
        state = {"messages": [msg]}
        result = greeter_router(state)
        self.assertEqual(result, "continue_greeter")
        
    def test_end_interaction_too_many_failures(self):
        fail_msg = ToolMessage(content="Incorrect answer", tool_call_id="1", name="verify_answer")
        state = {"messages": [fail_msg, fail_msg, fail_msg]}
        result = greeter_router(state)
        self.assertEqual(result, "end_interaction")

if __name__ == '__main__':
    unittest.main()
