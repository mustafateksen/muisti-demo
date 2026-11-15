"""Whether the live question box can be used (the key is set and live answers are not switched off)."""
import json, os
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        ok = bool(os.environ.get("VERDA_KEY")) and os.environ.get("MUISTI_LIVE", "on") != "off"
        origin = self.headers.get("Origin", "")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if origin.startswith(("https://mustafateksen.github.io", "http://127.0.0.1", "http://localhost")) or origin.endswith(".vercel.app"):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": ok}).encode())
