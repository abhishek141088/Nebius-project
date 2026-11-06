"""Minimal raw-TCP throughput test between the two worker nodes, over the
same eth0 path NCCL uses. Rank 0 is the server, rank 1 is the client -
driven by RANK/WORLD_SIZE env vars the same way torchrun sets them, so it
can reuse the exact same sbatch/srun launch pattern as everything else.
"""
import os
import socket
import time

PORT = 54321
CHUNK = 64 * 1024 * 1024  # 64 MiB send buffer, reused
DURATION_SECONDS = 8


def run_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print("[server] listening, waiting for client...")
    conn, addr = srv.accept()
    print(f"[server] connection from {addr}")

    total_bytes = 0
    start = time.monotonic()
    while True:
        data = conn.recv(CHUNK)
        if not data:
            break
        total_bytes += len(data)
    elapsed = time.monotonic() - start

    gbps = (total_bytes * 8) / elapsed / 1e9
    print(f"[server] received {total_bytes / 1e9:.2f} GB in {elapsed:.2f}s -> {gbps:.2f} Gbps")
    conn.close()
    srv.close()


def run_client(server_host):
    # Give the server a moment to start listening.
    time.sleep(3)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, PORT))
    print(f"[client] connected to {server_host}:{PORT}")

    buf = b"\x00" * CHUNK
    total_bytes = 0
    start = time.monotonic()
    while time.monotonic() - start < DURATION_SECONDS:
        sock.sendall(buf)
        total_bytes += len(buf)
    elapsed = time.monotonic() - start

    gbps = (total_bytes * 8) / elapsed / 1e9
    print(f"[client] sent {total_bytes / 1e9:.2f} GB in {elapsed:.2f}s -> {gbps:.2f} Gbps")
    sock.close()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    if rank == 0:
        run_server()
    else:
        server_host = os.environ["MASTER_ADDR"]
        run_client(server_host)
