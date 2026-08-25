"""Resolve demo quota tiers from Hugging Face OAuth identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
import threading
from typing import Any, Callable


logger = logging.getLogger("faster_qwen3_tts.demo.entitlements")

_DEFAULT_UNLIMITED_ORGS = {"huggingfacem4"}
_profile_cache: dict[str, dict[str, Any]] = {}
_profile_cache_lock = threading.Lock()


@dataclass(frozen=True)
class Entitlement:
    tier: str
    is_pro: bool
    is_team_member: bool

    @property
    def unlimited(self) -> bool:
        return self.is_pro or self.is_team_member


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _unlimited_orgs() -> set[str]:
    """Return case-insensitive org names whose members have unlimited usage."""
    raw = " ".join(
        (
            os.environ.get("UNLIMITED_ORGS", ""),
            os.environ.get("DEMO_UNLIMITED_ORGS", ""),
        )
    )
    configured = {
        value.strip().lower()
        for value in raw.replace(",", " ").split()
        if value.strip()
    }
    return _DEFAULT_UNLIMITED_ORGS | configured


def _oauth_token(oauth_info: Any) -> str | None:
    token = _field(oauth_info, "access_token")
    if token is None:
        return None
    return str(token).strip() or None


def _profile_cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _whoami_via_token(token: str) -> dict[str, Any]:
    """Fetch and cache the authenticated Hub ``whoami-v2`` profile."""
    cache_key = _profile_cache_key(token)
    with _profile_cache_lock:
        cached = _profile_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import httpx

        response = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        response.raise_for_status()
        profile = response.json()
        if not isinstance(profile, dict):
            raise ValueError("whoami-v2 returned a non-object response")
    except Exception as exc:  # pragma: no cover - network/permission dependent
        logger.info("whoami-v2 profile lookup failed: %r", exc)
        return {}

    with _profile_cache_lock:
        _profile_cache[cache_key] = profile
    return profile


def _org_names(user: Any, profile: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    org_sources = (_field(user, "orgs", []) or [], profile.get("orgs", []) or [])
    for orgs in org_sources:
        for org in orgs:
            for key in ("preferred_username", "name", "fullname", "sub"):
                value = _field(org, key)
                if value:
                    names.add(str(value).lower())
    return names


def resolve_entitlement(
    oauth_info: Any,
    *,
    profile_lookup: Callable[[str], dict[str, Any]] | None = None,
) -> Entitlement:
    """Resolve ``free``, ``pro``, or ``team`` from a Space OAuth session."""
    user = _field(oauth_info, "user_info")
    token = _oauth_token(oauth_info)
    profile: dict[str, Any] = {}
    if token:
        profile = (profile_lookup or _whoami_via_token)(token)

    is_pro = bool(_field(user, "is_pro", False) or _field(profile, "isPro", False))
    is_team_member = bool(_unlimited_orgs() & _org_names(user, profile))
    tier = "pro" if is_pro else "team" if is_team_member else "free"
    return Entitlement(tier=tier, is_pro=is_pro, is_team_member=is_team_member)
