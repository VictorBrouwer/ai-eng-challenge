import unittest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph.routing import dispatcher, greeter_router, route_after_bouncer, route_after_bouncer_tools, route_after_specialist
from src.graph.state import State


class TestDispatcher(unittest.TestCase):

    def test_dispatcher_defaults_to_greeter(self):
        state = {"messages": [], "active_agent": "greeter"}
        result = dispatcher(state)
        self.assertEqual(result, "greeter")

    def test_dispatcher_routes_to_bouncer(self):
        state = {"messages": [], "active_agent": "bouncer"}
        result = dispatcher(state)
        self.assertEqual(result, "bouncer")

    def test_dispatcher_routes_to_specialist(self):
        state = {"messages": [], "active_agent": "specialist"}
        result = dispatcher(state)
        self.assertEqual(result, "specialist")

    def test_dispatcher_missing_active_agent_defaults_to_greeter(self):
        state = {"messages": []}
        result = dispatcher(state)
        self.assertEqual(result, "greeter")


class TestGreeterRouter(unittest.TestCase):
    
    def test_greeter_continue_greeter_no_messages(self):
        state = {"messages": []}
        result = greeter_router(state)
        self.assertEqual(result, "continue_greeter")

    def test_greeter_call_tool(self):
        # If last message has tool calls, call tool
        msg = AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])
        state = {"messages": [msg]}
        result = greeter_router(state)
        self.assertEqual(result, "call_tool")

    def test_greeter_go_to_bouncer_verified_last(self):
        msg = ToolMessage(content='{"status": "VERIFIED"}', tool_call_id="1", name="verify_answer")
        state = {"messages": [msg]}
        result = greeter_router(state)
        self.assertEqual(result, "go_to_bouncer")
        
    def test_greeter_go_to_bouncer_verified_history(self):
        # Verified earlier, now user says thanks
        msgs = [
            ToolMessage(content='{"status": "VERIFIED"}', tool_call_id="1", name="verify_answer"),
            AIMessage(content="Welcome"),
            HumanMessage(content="Thanks")
        ]
        state = {"messages": msgs}
        result = greeter_router(state)
        self.assertEqual(result, "go_to_bouncer")
        
    def test_greeter_end_interaction_too_many_failures(self):
        # 3 failures = end
        fail_msg = ToolMessage(content="Incorrect answer", tool_call_id="1", name="verify_answer")
        state = {"messages": [fail_msg, fail_msg, fail_msg]}
        result = greeter_router(state)
        self.assertEqual(result, "end_interaction")

    def test_greeter_continue_on_regular_message(self):
        # Regular AI message without tool calls or verification -> continue
        msgs = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello! How can I help you?"),
            HumanMessage(content="I need help with my account"),
        ]
        state = {"messages": msgs}
        result = greeter_router(state)
        self.assertEqual(result, "continue_greeter")


class TestBouncerRouter(unittest.TestCase):
    
    def test_bouncer_check_account_tool(self):
        msg = AIMessage(content="", tool_calls=[{"name": "check_account_status", "args": {}, "id": "1"}])
        state = {"messages": [msg]}
        result = route_after_bouncer(state)
        self.assertEqual(result, "call_tool")
        
    def test_bouncer_handoff_tool_goes_through_tools(self):
        # handoff_to_specialist also routes through bouncer_tools first
        msg = AIMessage(content="", tool_calls=[{"name": "handoff_to_specialist", "args": {}, "id": "1"}])
        state = {"messages": [msg]}
        result = route_after_bouncer(state)
        self.assertEqual(result, "call_tool")
        
    def test_bouncer_continue_on_text_response(self):
        # Text response (no tool calls) means wait for user input
        msg = AIMessage(content="You are a regular client. Please call +1112112112.")
        state = {"messages": [msg]}
        result = route_after_bouncer(state)
        self.assertEqual(result, "continue_bouncer")

    def test_bouncer_continue_after_user_followup(self):
        # After bouncer responds and user follows up, bouncer continues
        msgs = [
            AIMessage(content="What can I help you with today?"),
            HumanMessage(content="I need help with a transfer"),
            AIMessage(content="Let me assist you with that."),
        ]
        state = {"messages": msgs}
        result = route_after_bouncer(state)
        self.assertEqual(result, "continue_bouncer")


class TestBouncerToolsRouter(unittest.TestCase):

    def test_handoff_tool_routes_to_specialist(self):
        # After handoff_to_specialist tool executes, route to specialist
        msg = ToolMessage(content="Handing off to Specialist.", tool_call_id="1", name="handoff_to_specialist")
        state = {"messages": [msg]}
        result = route_after_bouncer_tools(state)
        self.assertEqual(result, "go_to_specialist")

    def test_regular_tool_returns_to_bouncer(self):
        # After check_account_status executes, return to bouncer
        msg = ToolMessage(content="Premium", tool_call_id="1", name="check_account_status")
        state = {"messages": [msg]}
        result = route_after_bouncer_tools(state)
        self.assertEqual(result, "return_to_bouncer")


class TestSpecialistRouter(unittest.TestCase):

    def test_specialist_call_tool(self):
        # Specialist wants to call route_to_expert tool
        msg = AIMessage(content="", tool_calls=[{"name": "route_to_expert", "args": {"category": "yacht_insurance"}, "id": "1"}])
        state = {"messages": [msg]}
        result = route_after_specialist(state)
        self.assertEqual(result, "call_tool")

    def test_specialist_continue_on_text_response(self):
        msg = AIMessage(content="You are being connected to Yacht & Marine Insurance.")
        state = {"messages": [msg]}
        result = route_after_specialist(state)
        self.assertEqual(result, "continue_specialist")

    def test_specialist_continue_after_conversation(self):
        msgs = [
            AIMessage(content="I'm the specialist. Tell me about your request."),
            HumanMessage(content="I need yacht insurance."),
            AIMessage(content="Routing you to our Yacht & Marine Insurance department."),
        ]
        state = {"messages": msgs}
        result = route_after_specialist(state)
        self.assertEqual(result, "continue_specialist")


if __name__ == '__main__':
    unittest.main()
