# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Custom loggers."""

import logging

# uvicorn_logger is created to print out the logs in the terminal
# when running the server with `uvicorn server.main:app ...`
uvicorn_logger = logging.getLogger("uvicorn.error")
