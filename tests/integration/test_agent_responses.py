"""
Agent response quality tests with real LLM invocations.

These tests focus on individual agent behaviors: whether they call the right
tools at the right time, produce appropriate responses, and correctly
transition between agents.

Run with:
    pytest tests/integration/test_agent_responses.py -v
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

pytestmark = pytest.mark.integration


# ═══════════════════════════════════════════════════════════════════════════
# Greeter Agent Behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestGreeterToolCalling:
    """Verify the greeter calls the right tools at the right time."""

    def test_asks_for_details_when_only_one_provided(self, conversation):
        """
        When only a single detail (e.g. name) is given, the greeter should
        ask for more details instead of calling lookup_customer.
        """
        response = conversation.send("Hello, my name is Lisa")

        assert not conversation.was_tool_called("lookup_customer"), (
            "Greeter should NOT call lookup_customer with only 1 detail"
        )
        # Should ask for phone or IBAN
        response_lower = response.lower()
        assert any(kw in response_lower for kw in ["phone", "iban", "number", "detail", "identify"]), (
            f"Greeter should ask for additional identification. Got: {response}"
        )

    def test_calls_lookup_with_two_details(self, conversation):
        """
        When 2+ details are provided, the greeter should immediately
        call lookup_customer.
        """
        response = conversation.send(
            "Hi, my name is Lisa and my phone number is +1122334455"
        )

        assert conversation.was_tool_called("lookup_customer"), (
            "Greeter should call lookup_customer with 2 details"
        )
        lookup_result = conversation.get_tool_result("lookup_customer")
        assert lookup_result and "Customer found" in lookup_result, (
            f"Lookup should find Lisa. Got: {lookup_result}"
        )

    def test_asks_secret_question_after_successful_lookup(self, conversation):
        """
        After a successful lookup, the greeter should ask the customer's
        secret question (not skip it or ask for more details).
        """
        response = conversation.send(
            "Hello, I'm Lisa, my IBAN is DE89370400440532013000"
        )

        assert conversation.was_tool_called("lookup_customer")
        # Lisa's secret question is about her dog
        assert "dog" in response.lower(), (
            f"Greeter should ask about Lisa's dog. Got: {response}"
        )

    def test_calls_verify_answer_after_user_responds(self, conversation):
        """
        When the user answers the secret question, the greeter should
        call verify_answer with the answer and customer details.
        """
        conversation.send("Hi, I'm Lisa, phone +1122334455")
        conversation.send("Yoda")

        assert conversation.was_tool_called("verify_answer"), (
            "Greeter should call verify_answer after user answers secret question"
        )
        verify_result = conversation.get_tool_result("verify_answer")
        assert verify_result and "VERIFIED" in verify_result

    def test_lookup_with_all_three_details(self, conversation):
        """
        When all 3 details are given at once, the greeter should still
        call lookup_customer and proceed normally.
        """
        response = conversation.send(
            "Hi, I'm Lisa, phone +1122334455, IBAN DE89370400440532013000"
        )

        assert conversation.was_tool_called("lookup_customer"), (
            "Greeter should call lookup_customer with all 3 details"
        )
        assert "dog" in response.lower(), (
            f"Should still ask the secret question. Got: {response}"
        )

    def test_lookup_fails_for_mismatched_details(self, conversation):
        """
        When the provided details don't match any customer, the lookup
        should fail and the greeter should inform the user.
        """
        response = conversation.send(
            "Hi, my name is Lisa and my phone number is +9999999999"
        )

        assert conversation.was_tool_called("lookup_customer")
        lookup_result = conversation.get_tool_result("lookup_customer")
        assert lookup_result and "not found" in lookup_result.lower(), (
            f"Lookup should fail for mismatched details. Got: {lookup_result}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Bouncer Agent Behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestBouncerBehavior:
    """Verify the bouncer correctly identifies account status and routes."""

    def _verify_customer(self, conversation, name, phone, answer):
        """Helper: verify a customer through the greeter."""
        conversation.send(f"Hi, my name is {name}, phone {phone}")
        conversation.send(answer)

    def test_identifies_premium_account(self, conversation):
        """After verifying Lisa, bouncer should detect Premium status."""
        self._verify_customer(conversation, "Lisa", "+1122334455", "Yoda")

        assert conversation.was_tool_called("check_account_status")
        account_result = conversation.get_tool_result("check_account_status")
        assert account_result == "Premium"

    def test_identifies_regular_account(self, conversation):
        """After verifying John, bouncer should detect Regular status."""
        self._verify_customer(conversation, "John Smith", "+1234567890", "Berlin")

        assert conversation.was_tool_called("check_account_status")
        account_result = conversation.get_tool_result("check_account_status")
        assert account_result == "Regular"

    def test_premium_bouncer_asks_how_to_help(self, conversation):
        """
        After identifying a premium account with no specific request mentioned,
        the bouncer should ask how it can help.
        """
        self._verify_customer(conversation, "Lisa", "+1122334455", "Yoda")

        response = conversation.last_response.lower()
        assert any(kw in response for kw in ["how can i help", "assist", "what can i", "help you"]), (
            f"Bouncer should ask premium customer how to help. Got: {conversation.last_response}"
        )

    def test_regular_bouncer_provides_support_number(self, conversation):
        """
        Regular client should receive the support department number.
        """
        self._verify_customer(conversation, "John Smith", "+1234567890", "Berlin")

        response = conversation.last_response
        assert "+1112112112" in response, (
            f"Bouncer should provide regular support number. Got: {response}"
        )

    def test_premium_high_value_triggers_handoff(self, conversation):
        """
        When a premium customer mentions a high-value request, the bouncer
        should use handoff_to_specialist.
        """
        self._verify_customer(conversation, "Lisa", "+1122334455", "Yoda")
        conversation.send("I need help with yacht insurance")

        assert conversation.was_tool_called("handoff_to_specialist"), (
            "Bouncer should hand off yacht insurance request"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Specialist Agent Behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestSpecialistRouting:
    """Verify the specialist routes to the correct expert departments."""

    def _get_to_specialist(self, conversation, request_text):
        """Helper: verify Lisa and request a high-value service."""
        conversation.send("Hi, I'm Lisa, phone +1122334455")
        conversation.send("Yoda")
        return conversation.send(request_text)

    def test_routes_yacht_insurance(self, conversation):
        """Yacht/boat request → yacht_insurance department."""
        response = self._get_to_specialist(
            conversation, "I need yacht insurance for my sailing vessel"
        )

        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "Yacht & Marine Insurance" in route_result
        assert "+1999888001" in response

    def test_routes_wealth_management(self, conversation):
        """Investment/portfolio request → wealth_management department."""
        response = self._get_to_specialist(
            conversation, "I'd like to discuss portfolio management and investments"
        )

        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "Wealth Management" in route_result
        assert "+1999888002" in response

    def test_routes_real_estate(self, conversation):
        """Property/real estate request → real_estate department."""
        response = self._get_to_specialist(
            conversation, "I want to buy a luxury apartment and need real estate services"
        )

        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "Real Estate" in route_result
        assert "+1999888003" in response

    def test_routes_general_premium(self, conversation):
        """
        A premium high-value request that doesn't fit yacht/wealth/real estate
        should go to general_premium.
        """
        response = self._get_to_specialist(
            conversation, "I need help setting up a private jet charter account"
        )

        assert conversation.was_tool_called("route_to_expert")
        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "department" in route_result.lower()
        # Should get one of the valid department phone numbers
        assert any(
            phone in response
            for phone in ["+1999888001", "+1999888002", "+1999888003", "+1999888004"]
        ), f"Should include a department phone number. Got: {response}"


# ═══════════════════════════════════════════════════════════════════════════
# Agent Transition Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentTransitions:
    """Verify that active_agent transitions correctly through the flow."""

    def test_starts_with_greeter(self, conversation):
        """First interaction should be handled by the greeter."""
        conversation.send("Hello")
        assert conversation.active_agent == "greeter"

    def test_transitions_to_bouncer_after_verification(self, conversation):
        """After successful verification, active_agent should be bouncer."""
        conversation.send("Hi, I'm Lisa, phone +1122334455")
        conversation.send("Yoda")
        assert conversation.active_agent == "bouncer"

    def test_transitions_to_specialist_after_handoff(self, conversation):
        """After bouncer handoff, active_agent should be specialist."""
        conversation.send("Hi, I'm Lisa, phone +1122334455")
        conversation.send("Yoda")
        conversation.send("I need yacht insurance")
        assert conversation.active_agent == "specialist"

    def test_stays_with_bouncer_for_regular_customer(self, conversation):
        """Regular customer stays with bouncer (no specialist handoff)."""
        conversation.send("Hi, I'm John Smith, phone +1234567890")
        conversation.send("Berlin")
        assert conversation.active_agent == "bouncer"

        conversation.send("I need help with something")
        assert conversation.active_agent == "bouncer"
