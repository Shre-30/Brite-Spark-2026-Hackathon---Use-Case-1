"""
Single point of contact with the LLM. Everything else in this project
calls `generate()` — if you need to swap Ollama for something else
(a hosted API, a different local runtime), this is the only file that
should need to change. Keeping this boundary clean is also what makes
the day-two requirement change survivable: retrieval, refusal, and
answer construction never talk to the model directly.

Default backend: Ollama (local, no API key).
    Install: https://ollama.com
    Pull a model: `ollama pull llama3.1:8b`  (or qwen2.5:7b-instruct, phi3:mini)
    Ollama must be running: `ollama serve` (usually auto-starts)
"""

import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")


def generate(prompt: str, temperature: float = 0.0) -> str:
    """Sends a single prompt to Ollama and returns the raw text response.

    temperature=0.0 by default: for a grounded-answer / refusal system,
    determinism matters more than variety. This is a judgment call worth
    stating explicitly in DECISIONS.md.
    """
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} with model '{MODEL}'. "
            f"Is `ollama serve` running and has the model been pulled? "
            f"(`ollama pull {MODEL}`)\nUnderlying error: {e}"
        ) from e
