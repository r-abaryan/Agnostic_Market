"""The YAML implicit-bool footgun (Phase-0 plan Step 4).

`no`/`off`/`yes`/`NO` (Norway!) must NOT become Python booleans at parse time — they stay
strings. Only canonical `true`/`false` resolve to bool. This is what keeps a string field
whose value happens to be `no` from silently turning into `False` before Pydantic ever sees it.
"""

from __future__ import annotations

from pathlib import Path

from agnostic_market.config.loader import load_yaml_layer


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "layer.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_ambiguous_bool_tokens_stay_strings(tmp_path: Path) -> None:
    data = load_yaml_layer(
        _write(
            tmp_path,
            "country_code: NO\nflag_a: no\nflag_b: off\nflag_c: yes\n",
        )
    )
    assert data["country_code"] == "NO"
    assert data["flag_a"] == "no"
    assert data["flag_b"] == "off"
    assert data["flag_c"] == "yes"


def test_canonical_booleans_still_resolve(tmp_path: Path) -> None:
    data = load_yaml_layer(_write(tmp_path, "enabled: true\ndisabled: false\n"))
    assert data["enabled"] is True
    assert data["disabled"] is False


def test_ints_and_null_still_resolve(tmp_path: Path) -> None:
    # We only stripped the bool resolver — other implicit types must be unaffected.
    data = load_yaml_layer(_write(tmp_path, "n: 42\nf: 3.5\nnothing: null\n"))
    assert data["n"] == 42
    assert data["f"] == 3.5
    assert data["nothing"] is None
