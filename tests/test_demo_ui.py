import re
from pathlib import Path


DEMO_HTML = Path(__file__).parents[1] / "demo" / "index.html"


def test_user_banner_does_not_implicitly_sign_out():
    html = DEMO_HTML.read_text(encoding="utf-8")
    user_pill = re.search(r'<[^>]+id="userPill"[^>]*>', html)

    assert user_pill is not None
    assert user_pill.group(0).startswith("<button")
    assert "/oauth/huggingface/logout" not in user_pill.group(0)
    assert 'id="accountSignOut"' in html
    assert 'href="/oauth/huggingface/logout"' in html
