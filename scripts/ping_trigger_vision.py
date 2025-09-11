import socket, json

PI_IP = "10.1.149.182" # remember this
PI_PORT = 40002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Option A: plain text
sock.sendto(b"TRIGGER", (PI_IP, PI_PORT))
