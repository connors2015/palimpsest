"""Length-prefixed pickle wire protocol shared by the socket transport.

Every message is: 4-byte big-endian length header, then a pickled dict. Small
and dependency-free — enough to carry weights, shard assignments, and deltas
between a coordinator and miner processes.
"""

import pickle
import socket
import struct


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket):
    header = _recv_exactly(sock, 4)
    (length,) = struct.unpack(">I", header)
    return pickle.loads(_recv_exactly(sock, length))
