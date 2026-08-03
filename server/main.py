# SPDX-License-Identifier: MIT

"""Main app module."""

from fastapi import FastAPI

from server.api import config, mcp_oauth, mesh, metrics, models, openai, services, settings, utils
from server.api.fallback import StaticFilesHandler
from server.error_handlers import register_exception_handlers
from server.lifecycle import lifespan
from server.websockets import api as websocket

app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)
app.include_router(services.router)
app.include_router(settings.router)
app.include_router(config.router)
app.include_router(models.router)
app.include_router(websocket.router)
app.include_router(openai.router)
app.include_router(mesh.router)
app.include_router(metrics.router)
app.include_router(utils.router)
app.include_router(mcp_oauth.router)
# Deliberately kept as a separate router object (no auth dependency) — see module docstring.
app.include_router(mcp_oauth.callback_router)

app.mount("/", StaticFilesHandler(directory="static", html=True), name="static")
