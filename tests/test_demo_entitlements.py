from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace


_MODULE_PATH = Path(__file__).parents[1] / "demo" / "entitlements.py"
_SPEC = spec_from_file_location("demo_entitlements", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
entitlements = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = entitlements
_SPEC.loader.exec_module(entitlements)


def _oauth(*, user, token="oauth-token"):
    return SimpleNamespace(user_info=user, access_token=token)


def test_free_user_keeps_daily_limit():
    result = entitlements.resolve_entitlement(
        _oauth(user=SimpleNamespace(is_pro=False, orgs=[])),
        profile_lookup=lambda _token: {"isPro": False, "orgs": []},
    )

    assert result.tier == "free"
    assert result.unlimited is False


def test_pro_status_falls_back_to_whoami_v2():
    result = entitlements.resolve_entitlement(
        _oauth(user=SimpleNamespace(is_pro=False, orgs=[])),
        profile_lookup=lambda _token: {"isPro": True, "orgs": []},
    )

    assert result.tier == "pro"
    assert result.is_pro is True
    assert result.unlimited is True


def test_huggingface_m4_member_is_unlimited():
    result = entitlements.resolve_entitlement(
        _oauth(user=SimpleNamespace(is_pro=False, orgs=[])),
        profile_lookup=lambda _token: {
            "isPro": False,
            "orgs": [{"name": "HuggingFaceM4"}],
        },
    )

    assert result.tier == "team"
    assert result.is_team_member is True
    assert result.unlimited is True


def test_extra_unlimited_orgs_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("DEMO_UNLIMITED_ORGS", "Speech-Team, another-team")
    user = SimpleNamespace(
        is_pro=False,
        orgs=[SimpleNamespace(preferred_username="SPEECH-TEAM")],
    )
    result = entitlements.resolve_entitlement(
        _oauth(user=user, token=""),
        profile_lookup=lambda _token: {},
    )

    assert result.tier == "team"
    assert result.unlimited is True
