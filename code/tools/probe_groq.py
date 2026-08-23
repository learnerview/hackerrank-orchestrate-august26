"""One-shot Groq completion probe: prints the exact error or a sample response.
Does not reveal the API key. Run from code/: python tools/probe_groq.py
"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_env, Settings

load_env()
s = Settings()
from groq import Groq

client = Groq(api_key=s.groq_api_key, base_url=s.groq_base_url)
print(f"model_text={s.groq_model_text} base_url={s.groq_base_url} key={s.groq_api_key[:6]}...{s.groq_api_key[-4:]}")
try:
    r = client.chat.completions.create(
        model=s.groq_model_text,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": 'Reply with the JSON object {"ok": true}'}],
        max_tokens=20,
    )
    print("SUCCESS:", r.choices[0].message.content[:100])
except Exception as e:
    print("FAILED:", type(e).__name__, "|", str(e)[:400])
