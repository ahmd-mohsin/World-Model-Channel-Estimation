"""Serve the SSWM training dashboard over HTTP.

    python dashboard/serve.py [port]      # default 8000

Open http://localhost:<port>/ . The page polls metrics.json / eval.json (written live by
scripts/train-e2e.py) every 5 s. Run from the repo root or the dashboard dir.
"""

import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"SSWM dashboard at http://localhost:{PORT}/  (Ctrl-C to stop)", flush=True)
    httpd.serve_forever()
