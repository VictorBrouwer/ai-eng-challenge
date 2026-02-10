"""
End-to-end conversation flow tests with real LLM invocations.

These tests verify complete customer journeys through the multi-agent system:
greeter → bouncer → specialist, including tool calls, state transitions,
and response quality at each stage.

Customer data used (from data/customers.json):
  - Lisa:        phone +1122334455, IBAN DE89370400440532013000, Premium,  secret: pet's name = Yoda
  - John Smith:  phone +1234567890, IBAN DE89370400440532013001, Regular,  secret: birth city = Berlin
  - Maria Garcia: phone +9876543210, IBAN ES9121000418450200051332, Premium, secret: mother's maiden name = Rodriguez

Run with:
    pytest tests/integration/test_full_conversation_flows.py -v
"""

import pytest

pytestmark = pytest.mark.integration


# ═══════════════════════════════════════════════════════════════════════════
# Premium Customer → Specialist Routing
# ═══════════════════════════════════════════════════════════════════════════


class TestPremiumYachtInsuranceJourney:
    """Lisa (premium) → verify → bouncer → yacht insurance → specialist."""

    def test_full_flow(self, conversation):
        """
        Complete journey: greeter verifies Lisa, bouncer sees Premium,
        user requests yacht insurance, specialist routes to Yacht & Marine dept.
        """
        # ── Turn 1: Provide 2 identity details ──────────────────────────
        response = conversation.send(
            "Hello, my name is Lisa and my phone number is +1122334455"
        )

        assert conversation.was_tool_called("lookup_customer"), (
            "Greeter should call lookup_customer when given 2+ details"
        )
        # Greeter should ask the secret question about Lisa's pet
        assert "pet" in response.lower(), (
            f"Greeter should ask about Lisa's pet. Got: {response}"
        )

        # ── Turn 2: Answer secret question → verify → bouncer ──────────
        response = conversation.send("Yoda")

        assert conversation.was_tool_called("verify_answer"), (
            "Greeter should call verify_answer with the user's answer"
        )
        verify_result = conversation.get_tool_result("verify_answer")
        assert verify_result and "VERIFIED" in verify_result, (
            f"Lisa's verification should succeed. Got: {verify_result}"
        )

        # Bouncer should have checked account status after verification
        assert conversation.was_tool_called("check_account_status"), (
            "Bouncer should check account status after verification"
        )
        account_result = conversation.get_tool_result("check_account_status")
        assert account_result == "Premium", (
            f"Lisa's account should be Premium. Got: {account_result}"
        )
        assert conversation.active_agent == "bouncer"

        # ── Turn 3: Request yacht insurance → specialist routing ────────
        response = conversation.send(
            "I would like to get yacht insurance for my new boat"
        )

        assert conversation.was_tool_called("handoff_to_specialist"), (
            "Bouncer should hand off a yacht insurance request to specialist"
        )
        assert conversation.was_tool_called("route_to_expert"), (
            "Specialist should call route_to_expert"
        )

        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "Yacht & Marine Insurance" in route_result, (
            f"Should route to yacht insurance dept. Got: {route_result}"
        )

        # Final response must mention the department and contact number
        assert any(kw in response.lower() for kw in ["yacht", "marine"]), (
            f"Response should mention yacht/marine. Got: {response}"
        )
        assert "+9876543" in response, (
            f"Response should include +9876543. Got: {response}"
        )
        assert conversation.active_agent == "specialist"


class TestPremiumWealthManagementJourney:
    """Maria (premium) → verify → bouncer → wealth management → specialist."""

    def test_full_flow(self, conversation):
        # ── Turn 1: Maria provides name + IBAN ──────────────────────────
        response = conversation.send(
            "Hi, I'm Maria Garcia. My IBAN is ES9121000418450200051332"
        )

        assert conversation.was_tool_called("lookup_customer")
        assert any(kw in response.lower() for kw in ["maiden", "mother"]), (
            f"Should ask about mother's maiden name. Got: {response}"
        )

        # ── Turn 2: Answer secret question ──────────────────────────────
        response = conversation.send("Rodriguez")

        verify_result = conversation.get_tool_result("verify_answer")
        assert verify_result and "VERIFIED" in verify_result
        assert conversation.was_tool_called("check_account_status")

        # ── Turn 3: Request wealth management ───────────────────────────
        response = conversation.send(
            "I'd like to discuss wealth management and portfolio advisory services"
        )

        assert conversation.was_tool_called("route_to_expert")
        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "Wealth Management" in route_result
        assert "+1999888" in response, (
            f"Should include wealth management phone. Got: {response}"
        )


class TestPremiumRealEstateJourney:
    """Lisa (premium) via name+IBAN → verify → bouncer → real estate → specialist."""

    def test_full_flow(self, conversation):
        # ── Turn 1: Lisa provides name + IBAN ───────────────────────────
        response = conversation.send(
            "Hello, my name is Lisa and my IBAN is DE89370400440532013000"
        )
        assert conversation.was_tool_called("lookup_customer")

        # ── Turn 2: Answer secret question ──────────────────────────────
        response = conversation.send("Yoda")
        assert "VERIFIED" in (conversation.get_tool_result("verify_answer") or "")

        # ── Turn 3: Request real estate services ────────────────────────
        response = conversation.send(
            "I'm looking to purchase a luxury property and need real estate assistance"
        )

        assert conversation.was_tool_called("route_to_expert")
        route_result = conversation.get_tool_result("route_to_expert")
        assert route_result and "Real Estate" in route_result
        assert "+888666" in response, (
            f"Should include real estate phone. Got: {response}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Regular Customer Flow
# ═══════════════════════════════════════════════════════════════════════════


class TestRegularCustomerJourney:
    """John Smith (regular) → verify → bouncer → support number."""

    def test_regular_customer_gets_support_number(self, conversation):
        """John verifies and is directed to the regular support line."""
        # ── Turn 1: Provide details ─────────────────────────────────────
        response = conversation.send(
            "Hi, I'm John Smith. My phone number is +1234567890"
        )
        assert conversation.was_tool_called("lookup_customer")
        assert any(kw in response.lower() for kw in ["city", "born"]), (
            f"Should ask about birth city. Got: {response}"
        )

        # ── Turn 2: Answer secret question ──────────────────────────────
        response = conversation.send("Berlin")

        assert "VERIFIED" in (conversation.get_tool_result("verify_answer") or "")
        assert conversation.was_tool_called("check_account_status")

        account_result = conversation.get_tool_result("check_account_status")
        assert account_result == "Regular", (
            f"John should be Regular. Got: {account_result}"
        )

        # Bouncer should mention the regular support number
        assert "+1112112112" in response, (
            f"Should provide regular support number. Got: {response}"
        )
        assert conversation.active_agent == "bouncer"

    def test_regular_customer_cannot_access_specialist(self, conversation):
        """
        Even if a regular customer asks for high-value services, they should
        NOT be handed off to the specialist. They should be directed to the
        regular support line instead.
        """
        # Verify John Smith
        conversation.send("Hi, I'm John Smith, phone +1234567890")
        conversation.send("Berlin")

        # Request a high-value service
        response = conversation.send("I want yacht insurance for my boat")

        assert not conversation.was_tool_called("handoff_to_specialist"), (
            "Regular customers must NOT be handed off to specialist"
        )
        assert not conversation.was_tool_called("route_to_expert"), (
            "Specialist routing should not occur for regular customers"
        )
        # Should still direct to the regular support number
        assert "+1112112112" in response, (
            f"Should direct to regular support. Got: {response}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Premium Customer with General (Non-Specialist) Request
# ═══════════════════════════════════════════════════════════════════════════


class TestPremiumGeneralRequest:
    """Premium customer with a simple question → premium support, no specialist."""

    def test_general_request_gets_premium_support_number(self, conversation):
        """Lisa asks a general question and gets the premium support number."""
        # Verify Lisa
        conversation.send("Hi, I'm Lisa, phone +1122334455")
        conversation.send("Yoda")
        assert conversation.active_agent == "bouncer"

        # General (non-high-value) request
        response = conversation.send(
            "I have a question about my recent account statement"
        )

        assert not conversation.was_tool_called("handoff_to_specialist"), (
            "General requests should not trigger specialist handoff"
        )
        assert "+99887766" in response, (
            f"Should provide premium support number +99887766. Got: {response}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Failed Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestFailedVerification:
    """Identity verification failure after 3 incorrect answers."""

    def test_three_wrong_answers_ends_interaction(self, conversation):
        """
        After 3 incorrect secret-question answers, the system should end
        the interaction without ever reaching the bouncer.
        """
        # ── Turn 1: Provide details → get secret question ───────────────
        response = conversation.send("Hi, I'm Lisa, phone +1122334455")
        assert conversation.was_tool_called("lookup_customer")

        # ── Turns 2–4: Three wrong answers ──────────────────────────────
        for i in range(3):
            response = conversation.send(f"WrongAnswer{i + 1}")

        # verify_answer should have been called 3 times, all failures
        assert conversation.count_tool_calls("verify_answer") >= 3, (
            f"Expected ≥3 verify_answer calls, got {conversation.count_tool_calls('verify_answer')}"
        )
        verify_results = conversation.get_all_tool_results("verify_answer")
        failed_results = [r for r in verify_results if "VERIFIED" not in r]
        assert len(failed_results) >= 3, (
            f"Expected ≥3 failed verifications. Results: {verify_results}"
        )

        # Bouncer should never have been reached
        assert not conversation.was_tool_called("check_account_status"), (
            "Bouncer should not be reached after failed verification"
        )

    def test_wrong_then_correct_answer_succeeds(self, conversation):
        """
        One wrong answer followed by the correct answer should still succeed.
        """
        conversation.send("Hi, I'm Lisa, phone +1122334455")

        # Wrong answer first
        conversation.send("WrongAnswer")

        # Correct answer
        response = conversation.send("Yoda")

        # Should eventually verify
        verify_results = conversation.get_all_tool_results("verify_answer")
        assert any("VERIFIED" in r for r in verify_results), (
            f"Should have a successful verification. Results: {verify_results}"
        )

        # Bouncer should have been reached
        assert conversation.was_tool_called("check_account_status"), (
            "Bouncer should be reached after successful verification"
        )
