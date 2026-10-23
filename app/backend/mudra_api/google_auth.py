"""Google OpenID Connect sign-in for the SignLine web app."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token

router = APIRouter(prefix="/auth", tags=["authentication"])
SESSION_COOKIE = "signline_session"
OAUTH_COOKIE = "signline_oauth"
SESSION_TTL = 60 * 60 * 24 * 14
OAUTH_TTL = 10 * 60
_EPHEMERAL_SECRET = secrets.token_bytes(32)


def _secret() -> bytes:
    configured = os.getenv("SIGNLINE_AUTH_SECRET", "")
    return configured.encode("utf-8") if configured else _EPHEMERAL_SECRET


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _seal(payload: dict) -> str:
    encoded = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64(hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def _open(value: str | None, max_age: int) -> dict | None:
    if not value or "." not in value:
        return None
    encoded, signature = value.rsplit(".", 1)
    expected = _b64(hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_unb64(encoded))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("exp", 0)) < int(time.time()):
        return None
    if int(payload.get("iat", 0)) > int(time.time()) + 30 or int(payload["exp"]) - int(payload["iat"]) > max_age:
        return None
    return payload


def _redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:3000/auth/google/callback")


def _credentials() -> tuple[str, str] | None:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    return (client_id, client_secret) if client_id and client_secret else None


def _cookie_secure() -> bool:
    configured = os.getenv("SIGNLINE_AUTH_COOKIE_SECURE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return _redirect_uri().lower().startswith("https://")


def _cookie(response, name: str, value: str, *, max_age: int, path: str = "/") -> None:
    response.set_cookie(
        name,
        value,
        max_age=max_age,
        path=path,
        secure=_cookie_secure(),
        httponly=True,
        samesite="lax",
    )


def _safe_next(value: str | None) -> str:
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/app/home"
    if value.startswith("/admin"):
        return "/admin"
    if value.startswith("/app"):
        return value
    return "/app/home"


def _popup_result_response(status: str, next_path: str = "/app/home") -> HTMLResponse:
    success = status == "success"
    safe_next = _safe_next(next_path)
    payload = json.dumps(
        {"type": "signline-google-auth", "status": status, "next": safe_next},
        separators=(",", ":"),
    ).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    fallback = json.dumps(safe_next if success else f"/signin?error=google_{status}").replace("<", "\\u003c")
    response = HTMLResponse(
        f"""
        <html><head><title>SignLine sign-in</title></head>
        <body style="font-family:sans-serif;display:grid;place-items:center;height:100vh;margin:0;background:#f6f5ef;">
          <div style="text-align:center;max-width:420px;padding:24px;border:2px solid #1f1b1d;border-radius:18px;background:#fff;">
            <p style="margin:0 0 12px;font-weight:700;">{"Sign-in complete" if success else "Sign-in was not completed"}</p>
            <p>You can close this window.</p>
            <script>
              const result = {payload};
              try {{
                if ("BroadcastChannel" in window) {{
                  const channel = new BroadcastChannel("signline-google-auth");
                  channel.postMessage(result);
                  channel.close();
                }}
              }} catch {{}}
              try {{
                if (window.opener) window.opener.postMessage(result, window.location.origin);
              }} catch {{}}
              window.close();
              window.setTimeout(() => window.location.replace({fallback}), 500);
            </script>
          </div>
        </body></html>
        """
    )
    response.delete_cookie(
        OAUTH_COOKIE,
        path="/auth/google/callback",
        secure=_cookie_secure(),
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/me")
def me(request: Request) -> dict:
    session = _open(request.cookies.get(SESSION_COOKIE), SESSION_TTL)
    if not session:
        return {"user": None}
    return {"user": {key: session[key] for key in ("email", "name", "picture", "role") if key in session}}


@router.post("/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/", secure=_cookie_secure(), httponly=True, samesite="lax")
    return response


@router.get("/google/start")
def google_start(next: str = "/app/home"):
    credentials = _credentials()
    if not credentials:
        return RedirectResponse("/signin?error=google_not_configured", status_code=303)

    client_id, _ = credentials
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(48)
    challenge = _b64(hashlib.sha256(verifier.encode("ascii")).digest())
    now = int(time.time())
    oauth_state = _seal({
        "state": state,
        "nonce": nonce,
        "verifier": verifier,
        "next": _safe_next(next),
        "iat": now,
        "exp": now + OAUTH_TTL,
    })
    params = {
        "client_id": client_id,
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }
    response = RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params), status_code=302)
    _cookie(response, OAUTH_COOKIE, oauth_state, max_age=OAUTH_TTL, path="/auth/google/callback")
    return response

@router.get("/google/callback")
def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    oauth = _open(request.cookies.get(OAUTH_COOKIE), OAUTH_TTL)
    if error or not oauth or not state or not hmac.compare_digest(state, str(oauth.get("state", ""))):
        return _popup_result_response("cancelled")

    credentials = _credentials()
    if not credentials or not code:
        return _popup_result_response("failed")
    client_id, client_secret = credentials
    failure_stage = "token_exchange"
    try:
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
                "code_verifier": oauth["verifier"],
            },
            timeout=15,
        )
        token_response.raise_for_status()
        id_token_value = token_response.json().get("id_token")
        failure_stage = "id_token_verification"
        claims = id_token.verify_oauth2_token(
            id_token_value,
            GoogleRequest(),
            audience=client_id,
            clock_skew_in_seconds=60,
        )
        failure_stage = "identity_claim_validation"
        if claims.get("nonce") != oauth["nonce"] or not claims.get("email_verified"):
            raise ValueError("Google identity token did not match the sign-in request.")
        email = str(claims.get("email", "")).strip().lower()
        subject = str(claims.get("sub", ""))
        if not email or not subject:
            raise ValueError("Google did not return a verified email address.")
    except (requests.RequestException, GoogleAuthError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # Never log authorization codes, tokens, or identity claims.
        from logging import getLogger
        detail = ""
        if failure_stage == "id_token_verification" and isinstance(exc, GoogleAuthError):
            message = str(exc).lower()
            if "audience" in message:
                detail = " (audience mismatch)"
            elif "expired" in message or "used too early" in message:
                detail = " (token time validation)"
            else:
                detail = " (token signature or format validation)"
        getLogger("mudra_api.auth").warning(
            "Google sign-in failed during %s: %s%s",
            failure_stage,
            type(exc).__name__,
            detail,
        )
        return _popup_result_response("failed")

    admins = {entry.strip().lower() for entry in os.getenv("SIGNLINE_ADMIN_EMAILS", "").split(",") if entry.strip()}
    role = "admin" if email in admins else "user"
    now = int(time.time())
    session_cookie = _seal({
        "sub": subject,
        "email": email,
        "name": str(claims.get("name") or email.split("@", 1)[0])[:100],
        "role": role,
        "iat": now,
        "exp": now + SESSION_TTL,
    })
    next_path = oauth.get("next", "/app/home")
    if role != "admin" and next_path == "/admin":
        next_path = "/app/home"
    response = _popup_result_response("success", next_path)
    _cookie(response, SESSION_COOKIE, session_cookie, max_age=SESSION_TTL)
    return response
