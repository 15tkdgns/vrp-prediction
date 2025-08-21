#!/usr/bin/env python3
"""
Simple HTTP server for local development to avoid CORS issues
Enhanced with RSS feed proxy support and API key management
"""
import http.server
import socketserver
import os
import sys
import urllib.request
import urllib.parse
import json
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

PORT = 8090


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Handle API keys endpoint
        if self.path == "/api/keys":
            try:
                api_keys = {
                    "alphaVantage": os.getenv("ALPHA_VANTAGE_KEY"),
                    "financialModelingPrep": os.getenv("FMP_KEY"),
                    "twelveData": os.getenv("TWELVE_DATA_KEY"),
                    "polygon": os.getenv("POLYGON_KEY"),
                    "iexCloud": os.getenv("IEX_CLOUD_KEY"),
                    "marketaux": os.getenv("MARKETAUX_KEY"),
                }

                # Filter out None values
                api_keys = {k: v for k, v in api_keys.items() if v is not None}

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(api_keys).encode("utf-8"))
                return

            except Exception as e:
                print(f"API Keys Error: {e}")
                self.send_error(500, f"API Keys error: {str(e)}")
                return

        # Handle RSS proxy requests
        elif self.path.startswith("/proxy/rss?"):
            try:
                # Parse the query parameters
                query_string = self.path.split("?", 1)[1]
                params = urllib.parse.parse_qs(query_string)
                url = params.get("url", [""])[0]

                if not url:
                    self.send_error(400, "Missing URL parameter")
                    return

                # Fetch the RSS feed
                with urllib.request.urlopen(url) as response:
                    content = response.read().decode("utf-8")

                # Send the response
                self.send_response(200)
                self.send_header("Content-type", "application/xml")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))

            except Exception as e:
                print(f"RSS Proxy Error: {e}")
                self.send_error(500, f"Proxy error: {str(e)}")
        else:
            # Handle regular file serving
            super().do_GET()


def main():
    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent.absolute()
    os.chdir(dashboard_dir)

    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            print(f"서버가 포트 {PORT}에서 시작되었습니다.")
            print(f"브라우저에서 http://localhost:{PORT} 접속하세요.")
            print(f"현재 디렉토리: {dashboard_dir}")
            print("Ctrl+C로 서버를 종료할 수 있습니다.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"서버 시작 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
