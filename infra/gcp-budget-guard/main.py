"""Stop one Compute Engine VM when its approved monthly budget is exceeded.

The function is deliberately narrow: it only reacts to budget messages for the
configured budget ID and only stops (never deletes) the configured instance.
Cloud Billing sends at-least-once notifications, so the terminated-state check
makes repeated messages safe.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

TARGET_PROJECT_ID = os.environ["TARGET_PROJECT_ID"]
TARGET_ZONE = os.environ["TARGET_ZONE"]
TARGET_INSTANCE = os.environ["TARGET_INSTANCE"]
EXPECTED_BUDGET_ID = os.environ["EXPECTED_BUDGET_ID"]
COST_LIMIT_INR = float(os.environ.get("COST_LIMIT_INR", "7000"))


def _budget_payload(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Decode the background-event shape delivered by Functions Framework."""
    raw_data = base64.b64decode(event["data"])
    return json.loads(raw_data.decode("utf-8")), event.get("attributes", {})


def stop_vm_on_budget(event: dict[str, Any], _context: Any) -> None:
    """Handle a Pub/Sub Cloud Billing budget notification."""
    payload, attributes = _budget_payload(event)
    if attributes.get("budgetId") != EXPECTED_BUDGET_ID:
        logging.info("Ignoring notification for another budget.")
        return
    if payload.get("currencyCode") != "INR":
        logging.warning("Ignoring non-INR budget notification: %s", payload.get("currencyCode"))
        return

    cost = float(payload.get("costAmount", 0))
    if cost < COST_LIMIT_INR:
        logging.info("Budget is below guard limit: ₹%.2f / ₹%.2f", cost, COST_LIMIT_INR)
        return

    # Keep the heavy Compute client out of idle/under-limit invocations.
    from google.cloud import compute_v1

    client = compute_v1.InstancesClient()
    instance = client.get(
        project=TARGET_PROJECT_ID,
        zone=TARGET_ZONE,
        instance=TARGET_INSTANCE,
    )
    if instance.status == "TERMINATED":
        logging.info("%s is already stopped.", TARGET_INSTANCE)
        return

    operation = client.stop(
        project=TARGET_PROJECT_ID,
        zone=TARGET_ZONE,
        instance=TARGET_INSTANCE,
    )
    operation.result(timeout=60)
    logging.warning(
        "Stopped %s because the approved monthly GCP budget reached ₹%.2f.",
        TARGET_INSTANCE,
        cost,
    )
