"""Fake OpenRouter for local tests: the first model always returns 429 (exercises fallback);
others stream back a short answer that echoes the context it was given."""
import json, time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
LAST = {}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    LAST["body"] = body
    if body["model"] == "google/gemma-4-31b-it:free":
        return JSONResponse({"error": {"message": "Rate limit exceeded: free-models-per-day", "code": 429}}, status_code=429)
    user = body["messages"][-1]["content"]
    ctx_lines = [l for l in user.splitlines() if l.startswith(("[", "WASALT"))][:4]
    text = "MOCK ANSWER from " + body["model"] + ". Context heads: " + " || ".join(c[:90] for c in ctx_lines)
    def gen():
        for i in range(0, len(text), 25):
            chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body["model"],
                     "choices": [{"index": 0, "delta": {"content": text[i:i+25]}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/last")
def last():
    return LAST.get("body", {})
