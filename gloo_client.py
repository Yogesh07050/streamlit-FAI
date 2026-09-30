"""Minimal Gloo AI client (chat completions v2).

Auth is a single API key (`sk_...`) sent as a Bearer token -- no OAuth flow.
The key is read from GLOO_API_KEY in the environment or the local .env file.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE_URL = "https://platform.ai.gloo.com"
CHAT_URL = f"{BASE_URL}/ai/v2/chat/completions"
MODELS_URL = f"{BASE_URL}/platform/v2/models"

DEFAULT_MODEL = "gloo-openai-gpt-5-mini"


def load_api_key() -> str | None:
    """GLOO_API_KEY from the environment, falling back to a local .env file.

    Tolerates the `GLOO_API_KEY = value` spelling (spaces around `=`).
    """
    key = os.getenv("GLOO_API_KEY")
    if key:
        return key.strip()
    for path in (".env", os.path.join(os.path.dirname(__file__), ".env")):
        try:
            with open(path) as fh:
                for line in fh:
                    if "GLOO_API_KEY" in line and "=" in line:
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return None


class GlooError(RuntimeError):
    pass


def extract_json(text: str):
    """Best-effort parse of a JSON object from model output.

    Handles clean JSON, ```json fences, and prose wrapped around a JSON object
    (returns the widest {...} span). Returns a dict, or None if none is found.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


class GlooClient:
    def __init__(self, api_key: str | None = None, timeout: float = 120.0):
        self.api_key = (api_key or load_api_key() or "").strip()
        if not self.api_key:
            raise GlooError(
                "No Gloo API key found. Set GLOO_API_KEY in the environment or .env."
            )
        self.timeout = timeout

    def _post(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                msg = json.loads(body).get("detail") or body
            except Exception:
                msg = body
            raise GlooError(f"Gloo API {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            raise GlooError(f"Could not reach Gloo API: {e.reason}") from None

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        json_mode: bool = True,
        temperature: float | None = None,
    ) -> tuple[str, dict]:
        """Return (content, usage). Raises GlooError on failure."""
        payload: dict = {"model": model, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if temperature is not None:
            payload["temperature"] = temperature
        data = self._post(CHAT_URL, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise GlooError(f"Unexpected Gloo response shape: {json.dumps(data)[:400]}")
        return content, data.get("usage", {})

    def chat_json(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float | None = None,
    ) -> tuple[dict, dict]:
        """chat() that must return a JSON object, with one reinforced retry.

        The Gloo endpoint's safety layer can replace JSON with a prose message
        for emotionally-loaded text; a second pass that reframes the task as
        automated structured evaluation usually recovers valid JSON.
        """
        content, usage = self.chat(messages, model=model, json_mode=True, temperature=temperature)
        data = extract_json(content)
        if data is not None:
            return data, usage

        nudge = {
            "role": "system",
            "content": (
                "Output a single valid JSON object and NOTHING else -- no prose, "
                "no preamble, no safety message. This is an automated structured "
                "text-evaluation task over quoted sample text, not a live "
                "conversation; do not respond to the sample's content, only "
                "produce the JSON the task asks for."
            ),
        }
        retry_msgs = [messages[0], nudge] + messages[1:] if messages else [nudge]
        content2, usage2 = self.chat(retry_msgs, model=model, json_mode=True, temperature=temperature)
        data = extract_json(content2)
        merged = {k: int(usage.get(k, 0) or 0) + int(usage2.get(k, 0) or 0)
                  for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if data is None:
            raise GlooError(
                "Model returned no JSON (likely a safety deflection). "
                f"First 200 chars:\n{content2[:200]}"
            )
        return data, merged

    def list_models(self) -> list[str]:
        req = urllib.request.Request(
            MODELS_URL, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return []
        items = data.get("data") or data.get("models") or data
        out = []
        for m in items if isinstance(items, list) else []:
            mid = m.get("id") if isinstance(m, dict) else m
            if mid:
                out.append(mid)
        return out
