"""Print the OpenRouter models that are currently free, so OPENROUTER_MODELS can be updated.
Free models rotate often. Usage: python scripts/list_free_models.py"""
import json
import urllib.request

with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
    models = json.load(r)["data"]

free = [m for m in models if m["id"].endswith(":free") or
        (str(m.get("pricing", {}).get("prompt")) == "0" and str(m.get("pricing", {}).get("completion")) == "0")]
free.sort(key=lambda m: -(m.get("context_length") or 0))
print(f"{len(free)} free models\n")
for m in free:
    print(f"{m['id']:<60} ctx={m.get('context_length') or '?':>7}")
print("\nSet the 2-3 strongest instruction-tuned ones (plus openrouter/free last) in OPENROUTER_MODELS.")
