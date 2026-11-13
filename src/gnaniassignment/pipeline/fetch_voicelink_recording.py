"""Fetch one VoiceLink call recording using its documented call-details API.

Set VOICELINK_ACCESS_TOKEN in the shell. The token is never printed or saved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import requests


CALL_DETAILS_URL = "https://app.voicelink.co.in/api/v1/call-log/details"
MAX_RECORDING_BYTES = 100 * 1024 * 1024


def recording_url_from_details(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("VoiceLink returned no call details")
    url = data.get("recording_url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("This call has no recording_url; check whether recording is enabled on its DID")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("VoiceLink returned an invalid recording URL")
    return url


def fetch(call_id: str, destination: Path, token: str) -> int:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(CALL_DETAILS_URL, params={"call_id": call_id}, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") is not True:
        raise ValueError("VoiceLink could not find this call in the authenticated account")
    url = recording_url_from_details(payload.get("data"))
    # A recording URL can belong to a storage provider. Never send the API
    # bearer token to that host.
    recording_headers = headers if urlparse(url).hostname == urlparse(CALL_DETAILS_URL).hostname else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with requests.get(url, headers=recording_headers, stream=True, timeout=30) as audio:
        audio.raise_for_status()
        with destination.open("xb") as output:
            os.chmod(destination, 0o600)
            try:
                for chunk in audio.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_RECORDING_BYTES:
                        raise ValueError("Recording exceeds the 100 MB download limit")
                    output.write(chunk)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("VoiceLink returned an empty recording")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-id", required=True, help="VoiceLink call_id or unique_id")
    parser.add_argument("--output", required=True, type=Path, help="New local file; must not already exist")
    args = parser.parse_args()
    token = os.getenv("VOICELINK_ACCESS_TOKEN", "").strip()
    if not token:
        parser.error("Set VOICELINK_ACCESS_TOKEN from a VoiceLink login before fetching")
    size = fetch(args.call_id, args.output, token)
    print(f"Saved {size:,} bytes to {args.output}")


if __name__ == "__main__":
    main()
