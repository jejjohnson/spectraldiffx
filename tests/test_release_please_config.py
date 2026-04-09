from __future__ import annotations

import json
from pathlib import Path


def test_release_please_uses_plain_semver_tags() -> None:
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "release-please-config.json").read_text()
    )

    assert config["include-component-in-tag"] is False
    assert config["include-v-in-tag"] is False
