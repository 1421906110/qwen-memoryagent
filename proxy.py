#!/usr/bin/env python3
"""
MemoryAgent — Local proxy to bypass system proxy for ECS access.

Usage:
    python proxy.py              # Start on default port 9999
    python proxy.py --port 8888  # Custom port

Then open http://127.0.0.1:9999/ in your browser.
"""
import http.server
import urllib.request
import urllib.error
import os
import sys

TARGET = os.environ.get("PROXY_TARGET", "http://47.99.151.253:8000")

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''
        self._proxy(body=body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def _proxy(self, body=None):
        url = TARGET + self.path
        try:
            req = urllib.request.Request(url, data=body, method=self.command)
            ct = self.headers.get('Content-Type')
            if ct:
                req.add_header('Content-Type', ct)

            # Bypass system proxy
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)

            resp = opener.open(req, timeout=60)
            data = resp.read()

            self.send_response(resp.status)
            self.send_header('Access-Control-Allow-Origin', '*')
            content_type = resp.headers.get('Content-Type', 'application/json')
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(f'{{"error":"{e}"}}'.encode())

    def log_message(self, format, *args):
        print(f"[{self.command} {self.path}] {args[0]}")

if __name__ == '__main__':
    port = 9999
    if '--port' in sys.argv:
        idx = sys.argv.index('--port')
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    server = http.server.HTTPServer(('127.0.0.1', port), ProxyHandler)
    print(f"🚀 MemoryAgent Local Proxy")
    print(f"   Local:  http://127.0.0.1:{port}/")
    print(f"   Target: {TARGET}")
    print(f"   Open in browser → 关掉系统代理即可正常使用")
    print(f"   Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Proxy stopped")
        server.server_close()