# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Port Service module."""

import socket


class PortService:
    def __init__(
        self,
    ):
        self.allocated_ports = set[int]()

    def get_free_port(self, start: int = 20_000, end: int = 30_000) -> int:
        """Get next free port."""
        for port in range(start, end + 1):
            if port in self.allocated_ports:
                continue
            if self.is_port_available(port):
                self.allocated_ports.add(port)
                return port
        raise RuntimeError("No free port in range", (start, end))

    def is_port_available(self, port: int) -> bool:
        """Check whether port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.close()
            return True  # noqa: TRY300
        except OSError:
            return False
