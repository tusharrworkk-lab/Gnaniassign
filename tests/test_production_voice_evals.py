"""Regression tests for real-telephony and sampled-production evals."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from gnaniassignment.telephony.production_evals import (
    BedrockRuleProposalGenerator,
    EVAL_CUSTOMER_PREFIX,
    SCENARIOS,
    VoiceLinkEvalClient,
    should_sample_session,
    targeted_review_reasons,
    validate_rule_proposal,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return Response({"status": True, "data": {"outbound_queue_id": 42, "accepted_leads": 1}})

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return Response({"status": True, "data": {"call_id": "call-1", "recording_url": "https://recording.invalid/call.mp3"}})


class BedrockClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return {"output": {"message": {"content": [{"text": json.dumps(self.payload)}]}}}


class ProductionVoiceEvalTests(unittest.TestCase):
    def test_scenarios_are_versioned_and_unique(self) -> None:
        identifiers = [scenario.id for scenario in SCENARIOS]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(all(identifier.endswith("-v1") for identifier in identifiers))
        self.assertTrue(any(scenario.persona == "personal_assistant" for scenario in SCENARIOS))
        self.assertTrue(any(scenario.persona == "corporate_assistant" for scenario in SCENARIOS))
        self.assertTrue(any(scenario.persona == "retail_assistant" for scenario in SCENARIOS))

    def test_real_call_uses_documented_india_number_shape_and_safe_correlation(self) -> None:
        session = Session()
        with patch.dict("os.environ", {"VOICELINK_EVAL_DID_NUMBER": "18001234567"}, clear=False):
            client = VoiceLinkEvalClient(access_token="secret", base_url="https://voice.test/api/v1", session=session)
            result = client.queue_eval_call("9876543210", "run-1", "scenario-1")
        self.assertEqual(result["outbound_queue_id"], 42)
        url, request = session.posts[0]
        self.assertEqual(url, "https://voice.test/api/v1/add_lead")
        self.assertEqual(request["json"]["customer_number"], "9876543210")
        self.assertEqual(request["json"]["country_code"], "91")
        custom = json.loads(request["json"]["custom_parameters"])
        self.assertEqual(custom["customer_id"], EVAL_CUSTOMER_PREFIX + "run-1")
        self.assertNotIn("secret", json.dumps(request["json"]))

    def test_real_call_rejects_prefixed_india_number(self) -> None:
        with patch.dict("os.environ", {"VOICELINK_EVAL_DID_NUMBER": "18001234567"}, clear=False):
            client = VoiceLinkEvalClient(access_token="secret", session=Session())
            with self.assertRaisesRegex(ValueError, "10 digits"):
                client.queue_eval_call("919876543210", "run-1", "scenario-1")

    def test_sampling_is_stable_and_bounded(self) -> None:
        self.assertEqual(should_sample_session("call-1", 0.25), should_sample_session("call-1", 0.25))
        self.assertFalse(should_sample_session("call-1", 0.0))
        self.assertTrue(should_sample_session("call-1", 1.0))

    def test_failed_delivery_is_always_targeted_for_review(self) -> None:
        reasons = targeted_review_reasons(
            [{"delivery_status": "interrupted"}],
            {"voice_quality_judgment": {"overall_pass": False, "needs_human_review": True, "critical_failures": ["fabricated action"]}},
        )
        self.assertEqual(reasons, ["judge_failed", "judge_requested_review", "critical_failure", "delivery_failure"])

    def test_call_details_uses_call_log_endpoint(self) -> None:
        session = Session()
        client = VoiceLinkEvalClient(access_token="secret", base_url="https://voice.test/api/v1", session=session)
        details = client.call_details("call-1")
        self.assertEqual(details["recording_url"], "https://recording.invalid/call.mp3")
        self.assertEqual(session.gets[0][1]["params"], {"call_id": "call-1"})

    def test_rule_generator_returns_inactive_candidate_shape(self) -> None:
        client = BedrockClient({
            "rule_kind": "dont",
            "rule_text": "Do not claim a message was delivered unless an approved tool confirms delivery.",
            "suggested_priority": 20,
            "reason": "The reviewed answer fabricated completion.",
        })
        proposal = BedrockRuleProposalGenerator(client=client, model_id="judge-test").propose({
            "decision": "never_say_this",
            "comment": "No delivery happened",
            "caller_utterance": "Please tell them",
            "assistant_speech_actually_handed_off": "I delivered it",
        })
        self.assertEqual(proposal["rule_kind"], "dont")
        self.assertEqual(proposal["suggested_priority"], 20)
        request_payload = json.loads(client.requests[0]["messages"][0]["content"][0]["text"])
        self.assertIn("assistant_speech_actually_handed_off", request_payload)

    def test_rule_proposal_is_bounded_and_validated(self) -> None:
        normalized = validate_rule_proposal(
            {"rule_kind": "do", "rule_text": "  Acknowledge   corrections clearly. ", "suggested_priority": 99999},
            "disapproved",
        )
        self.assertEqual(normalized["rule_text"], "Acknowledge corrections clearly.")
        self.assertEqual(normalized["suggested_priority"], 10000)
        with self.assertRaisesRegex(ValueError, "rule_kind"):
            validate_rule_proposal({"rule_text": "Missing kind"}, "disapproved")


if __name__ == "__main__":
    unittest.main()
