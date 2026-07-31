# SPDX-License-Identifier: MIT

"""Main app module."""

from fastapi import FastAPI

from server.api import config, mesh, metrics, models, openai, services, settings, utils
from server.api.fallback import StaticFilesHandler
from server.lifecycle import lifespan
from server.websockets import api as websocket

app = FastAPI(lifespan=lifespan)
app.include_router(services.router)
app.include_router(settings.router)
app.include_router(config.router)
app.include_router(models.router)
app.include_router(websocket.router)
app.include_router(openai.router)
app.include_router(mesh.router)
app.include_router(metrics.router)
app.include_router(utils.router)

app.mount("/", StaticFilesHandler(directory="static", html=True), name="static")
