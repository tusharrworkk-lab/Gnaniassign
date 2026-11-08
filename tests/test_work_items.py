"""Contract tests for the Plane adapter's read and approved-write boundary."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from gnaniassignment.work_items import PlaneClient, PlaneSettings


class Response:
    def __init__(self, body: object, ok: bool = True, status_code: int = 200) -> None:
        self.body, self.ok, self.status_code = body, ok, status_code

    def json(self):
        return self.body


class Session:
    def __init__(self, responses: list[Response]) -> None:
        self.responses, self.calls = responses, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class PlaneClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = PlaneSettings("https://api.plane.so", "tushar-work", "not-a-real-token")

    def test_list_work_items_uses_small_v2_projection(self):
        session = Session([Response({"data": [{"id": "item-1", "identifier": "APP-1", "name": "Ship approval queue"}]})])
        result = PlaneClient(self.settings, session).list_work_items(7)
        self.assertEqual(result[0]["identifier"], "APP-1")
        method, url, kwargs = session.calls[0]
        self.assertEqual(method, "GET")
        self.assertEqual(url, "https://api.plane.so/api/v2/workspaces/tushar-work/work-items/")
        self.assertEqual(kwargs["params"]["per_page"], "7")
        self.assertEqual(kwargs["headers"]["X-Api-Key"], "not-a-real-token")

    def test_approved_comment_is_html_escaped_and_tagged_for_deduplication(self):
        session = Session([
            Response({"data": []}),
            Response({"id": "comment-1"}),
        ])
        result = PlaneClient(self.settings, session).apply_comment(
            project_id="project-1", work_item_id="item-1", comment="Testing <blocked>", external_id="voice-agent:action-1"
        )
        self.assertEqual(result["id"], "comment-1")
        method, _, kwargs = session.calls[1]
        self.assertEqual(method, "POST")
        self.assertEqual(kwargs["json"]["comment_html"], "<p>Testing &lt;blocked&gt;</p>")
        self.assertEqual(kwargs["json"]["external_source"], "voice-agent")

    def test_existing_external_id_never_posts_a_duplicate_comment(self):
        session = Session([Response({"data": [{"id": "comment-existing"}]})])
        result = PlaneClient(self.settings, session).apply_comment(
            project_id="project-1", work_item_id="item-1", comment="already applied", external_id="voice-agent:action-1"
        )
        self.assertEqual(result["id"], "comment-existing")
        self.assertEqual(len(session.calls), 1)

    def test_environment_requires_a_workspace_and_secret(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PLANE_API_KEY"):
                PlaneSettings.from_environment()

