# SPDX-License-Identifier: MIT

"""Metrics API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from server.core.dependencies import auth_metrics, get_metrics_service
from server.metrics import MetricsService

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def metrics_endpoint(
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
    _: Annotated[None, Depends(auth_metrics)],
) -> Response:
    """Get Prometheus metrics."""
    return Response(
        content=metrics_service.get_current_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
