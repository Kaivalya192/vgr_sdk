#!/usr/bin/env python3
# ==================================
# FILE: scripts/trigger_client.py
# ==================================
import argparse
import json
import socket
import sys
import urllib.request

def send_udp(ip: str, port: int, payload: dict) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, (ip, port))
    sock.close()

def send_http(url: str, payload: dict, timeout: float = 2.0) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        _ = resp.read()

def main():
    ap = argparse.ArgumentParser(description="Send TRIGGER to Vision app")
    ap.add_argument("--mode", choices=["udp", "http"], default="udp", help="transport")
    ap.add_argument("--ip", default="127.0.0.1", help="UDP IP")
    ap.add_argument("--port", type=int, default=40011, help="UDP port for TRIGGER (your Vision app must listen)")
    ap.add_argument("--http-url", default="http://127.0.0.1:5000/trigger", help="HTTP trigger URL")
    ap.add_argument("--job-id", default="", help="optional job identifier")
    args = ap.parse_args()

    payload = {"cmd":"TRIGGER"}
    if args.job_id:
        payload["job_id"] = args.job_id

    try:
        if args.mode == "udp":
            send_udp(args.ip, args.port, payload)
            print(f"Sent UDP TRIGGER to {args.ip}:{args.port}")
        else:
            send_http(args.http_url, payload)
            print(f"Sent HTTP TRIGGER to {args.http_url}")
    except Exception as e:
        print(f"Trigger send failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
