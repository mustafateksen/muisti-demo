"""MUISTI demo: one question answered live over the demo memory.

The same three steps as the app's QueryEngine, cut down: embed the question, rank the stored
records by cosine, rerank the best ones, then let Nemotron answer from those records only.
The inference key comes from the environment (VERDA_KEY); it is never in the repository.
"""
import base64, json, os, re, struct, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.environ.get("MUISTI_CHAT_URL", "https://containers.datacrunch.io/muisti-nemotron/v1").rstrip("/")
RETRIEVAL = os.environ.get("MUISTI_RETRIEVAL_URL", "https://containers.datacrunch.io/muisti-retrieval").rstrip("/")
ALLOWED = ("https://mustafateksen.github.io", "http://127.0.0.1", "http://localhost")
_data = None
_hits = {}


def data():
    global _data
    if _data is None:
        for path in (os.path.join(HERE, "..", "data", "server.json"), os.path.join(HERE, "server.json"), "data/server.json"):
            if os.path.exists(path):
                raw = json.load(open(path, encoding="utf-8"))
                for r in raw["records"]:
                    blob = base64.b64decode(r.pop("vector"))
                    v = struct.unpack(f"<{len(blob) // 4}f", blob)
                    norm = sum(x * x for x in v) ** 0.5 or 1.0
                    r["v"] = [x / norm for x in v]
                _data = raw
                break
    return _data


def post(url, body, timeout=12):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Authorization", "Bearer " + os.environ.get("VERDA_KEY", ""))
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def embed(text):
    out = post(RETRIEVAL + "/v2/embed", {"model": "nemotron-3.5-embed", "input_type": "query", "embedding_types": ["float"], "truncate": "END",
                                         "inputs": [{"content": [{"type": "text", "text": text[:1000]}]}]})
    v = out["embeddings"]["float"][0]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def rerank_one(question, passage):
    out = post(RETRIEVAL + "/classify", {"model": "nemotron-3.5-rerank", "add_special_tokens": False, "use_activation": False,
                                         "messages": [{"role": "query", "content": [{"type": "text", "text": question[:500]}]},
                                                      {"role": "document", "content": [{"type": "text", "text": passage[:1200]}]}]})
    return out["data"][0]["probs"][0]


def passage(r):
    parts = [r["title"], r.get("caption") or "", ("Text: " + r["ocr"].replace("\n", " | ")) if r.get("ocr") else "", r["street"], r.get("area") or ""]
    return ". ".join(p for p in parts if p)


SYSTEM = """You are Muisti, a memory assistant that lives in a pair of smart glasses. The glasses saw things \
earlier today and kept them as memory records (a picture, the text read in it, a short description, time \
and place). The user now asks a question. Answer ONLY from the records you are given and from what the \
user told Muisti (listed below). If what they ask about is not in the records, say plainly that you did \
not see it; never guess, and never make up a name, number, time or place. A record is the most likely \
match when it is first, but none may match. Speak to the user as "you", in the language of the question, \
in one or two short sentences a person would say aloud, naming where and when it was seen when that helps.

Output JSON: {"label": "seen" when the records answer the question, "not_seen" when they do not, "unsure" \
when they only partly do; "answer": the spoken answer; "used_ids": the ids of the records the answer rests on}."""

SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["seen", "not_seen", "unsure"]}, "answer": {"type": "string"},
                                           "used_ids": {"type": "array", "items": {"type": "integer"}}}, "required": ["label", "answer", "used_ids"]}


def answer(question, records, told):
    listing = "\n\n".join(f"id: {r['id']}\nSeen: today {r['time']} at {r['place']} ({r['street']}, {r.get('area') or ''})\nWhat: {r['title']}\n"
                          f"Description: {r.get('caption') or ''}\nText read: {(r.get('ocr') or '(none)').replace(chr(10), ' | ')}" for r in records)
    user = f"Current local time: {time.strftime('%H:%M')}\nWhat the user told Muisti: {told}\n\nQuestion: {question}\n\nRecords (most relevant first):\n{listing}"
    out = post(CHAT + "/chat/completions", {"model": "nemotron-3.5-super-vl", "max_tokens": 300, "temperature": 0, "top_p": 0.95, "stream": False,
                                            "chat_template_kwargs": {"enable_thinking": False},
                                            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}},
                                            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}, timeout=20)
    text = out["choices"][0]["message"]["content"]
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def numbers(text):
    return set(re.findall(r"\d+(?:[.,:]\d+)*", text))


def ask(question):
    d = data()
    timings = {}
    started = time.time()
    q = embed(question)
    timings["embed"] = int((time.time() - started) * 1000)
    ranked = sorted(d["records"], key=lambda r: -sum(a * b for a, b in zip(q, r["v"])))[:10]
    started = time.time()
    with ThreadPoolExecutor(max_workers=10) as pool:
        scores = list(pool.map(lambda r: _score(question, r), ranked))
    timings["rerank"] = int((time.time() - started) * 1000)
    best = [r for _, r in sorted(zip(scores, ranked), key=lambda p: -p[0])][:5]
    started = time.time()
    reply = answer(question, best, d["told"])
    timings["answer"] = int((time.time() - started) * 1000)
    shown = {r["id"]: r for r in best}
    used = [i for i in reply.get("used_ids", []) if i in shown]
    label = reply.get("label", "unsure")
    text = (reply.get("answer") or "").strip()
    # The app's evidence rule, in short: a number in the answer must stand in a record that was shown.
    known = numbers(" ".join(passage(r) + " " + r["time"] for r in best) + " " + d["told"] + " " + question)
    if label == "seen" and not numbers(text) <= known:
        label, text, used = "unsure", "I'm not sure I saw that clearly, so I'd rather not guess.", []
    if label != "seen":
        used = []
    return {"answer": text, "label": label, "evidence": used[:3], "timings": timings}


def _score(question, r):
    try:
        return rerank_one(question, passage(r))
    except Exception:
        return float("-inf")


def allowed(ip):
    now = time.time()
    recent = [t for t in _hits.get(ip, []) if now - t < 60]
    if len(recent) >= 10:
        return False
    _hits[ip] = recent + [now]
    return True


class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        origin = self.headers.get("Origin", "")
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if origin.startswith(ALLOWED) or origin.endswith(".vercel.app"):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        ok = bool(os.environ.get("VERDA_KEY")) and os.environ.get("MUISTI_LIVE", "on") != "off" and data() is not None
        self._send(200, {"ok": ok})

    def do_POST(self):
        if os.environ.get("MUISTI_LIVE", "on") == "off" or not os.environ.get("VERDA_KEY"):
            return self._send(503, {"error": "live answers are switched off"})
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()
        if not allowed(ip):
            return self._send(429, {"error": "too many questions, try again in a minute"})
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 4000)
            question = str(json.loads(self.rfile.read(length) or b"{}").get("question", "")).strip()[:300]
        except Exception:
            return self._send(400, {"error": "bad request"})
        if len(question) < 2:
            return self._send(400, {"error": "no question"})
        try:
            self._send(200, ask(question))
        except Exception as error:
            self._send(502, {"error": "the models did not answer", "detail": type(error).__name__})
