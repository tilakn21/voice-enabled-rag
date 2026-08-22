"""Validate a Sarvam API key before wiring it in.

Sarvam has two different things called "a key":
  * a **Samvaad** key (prefix `sk_samvaad_`) — for the Samvaad conversational
    agent product. It does NOT authenticate the developer REST API.
  * an **API subscription key** — from the Sarvam dashboard. This is the one
    `api.sarvam.ai` accepts, in the `api-subscription-key` header.

Usage:
    python scripts/check_sarvam_key.py <key>
    python scripts/check_sarvam_key.py            # reads SARVAM_API_KEY / .env
"""
import os
import sys
import wave
import tempfile
import httpx

def tiny_wav() -> str:
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1s of silence
    return path

def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SARVAM_API_KEY", "")
    if not key:
        env = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env):
            for line in open(env):
                if line.startswith("SARVAM_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        print("no key given. usage: python scripts/check_sarvam_key.py <key>")
        return 2

    print(f"key: {key[:14]}…{key[-4:]}  ({len(key)} chars)")
    if key.startswith("sk_samvaad_"):
        print("\n  ⚠  This is a SAMVAAD key, not an API subscription key.")
        print("     Samvaad is Sarvam's conversational-agent product; its keys do")
        print("     not authenticate api.sarvam.ai. Get an API subscription key")
        print("     from the Sarvam dashboard instead.\n")

    path = tiny_wav()
    with open(path, "rb") as fh:
        r = httpx.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": key},
            files={"file": ("probe.wav", fh, "audio/wav")},
            data={"model": "saarika:v2.5", "language_code": "unknown"},
            timeout=30,
        )
    if r.status_code == 200:
        print(f"✅ VALID — speech-to-text responded 200")
        print(f"   {r.text[:200]}")
        print("\n   Put it in .env as SARVAM_API_KEY=… and restart the server.")
        return 0
    print(f"❌ REJECTED — HTTP {r.status_code}")
    print(f"   {r.text[:250]}")
    if r.status_code == 403:
        print("\n   403 = the key itself is not accepted (not a quota problem).")
        print("   Get an API subscription key from https://dashboard.sarvam.ai")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
