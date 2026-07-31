# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/error_handlers.py."""

from server.error_handlers import (
    _error_type_for_status,  # pyright: ignore[reportPrivateUsage]
    _message_and_param_from_detail,  # pyright: ignore[reportPrivateUsage]
    build_error_body,
)


def test_build_error_body_shape() -> None:
    assert build_error_body("bad field", "invalid_request_error", "model", "some_code") == {
        "error": {"message": "bad field", "type": "invalid_request_error", "param": "model", "code": "some_code"}
    }


def test_error_type_for_status_maps_known_statuses() -> None:
    assert _error_type_for_status(400) == "invalid_request_error"
    assert _error_type_for_status(401) == "authentication_error"
    assert _error_type_for_status(403) == "permission_error"
    assert _error_type_for_status(404) == "not_found_error"
    assert _error_type_for_status(429) == "rate_limit_error"


def test_error_type_for_status_defaults_to_invalid_request_error() -> None:
    assert _error_type_for_status(418) == "invalid_request_error"


def test_error_type_for_status_maps_server_errors_to_api_error() -> None:
    assert _error_type_for_status(500) == "api_error"
    assert _error_type_for_status(503) == "api_error"


def test_message_and_param_from_string_detail() -> None:
    assert _message_and_param_from_detail("Model not found") == ("Model not found", None)


def test_message_and_param_from_pydantic_error_list_detail() -> None:
    detail = [{"field": "model", "message": "field required"}, {"field": "input", "message": "field required"}]

    message, param = _message_and_param_from_detail(detail)

    assert param == "model"
    assert message == "model: field required; input: field required"


def test_message_and_param_from_dict_detail_falls_back_to_str() -> None:
    detail = {"warnings": ["image is outdated"]}

    message, param = _message_and_param_from_detail(detail)

    assert param is None
    assert message == str(detail)
