# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.websockets.models import AncestorInfo
from server.websockets.parent_infra_group import ParentInfraGroup


def _make_parent(url: str, enabled: bool = True, ancestors: list[str] | None = None, ws: object = None) -> MagicMock:
    p = MagicMock()
    p.parent_url = url
    p.enabled = enabled
    # Mirror real behaviour: InitResponse.ancestors starts with the direct parent's own URL
    p.ancestors = [AncestorInfo(url=u, name="") for u in [url, *(ancestors or [])]]
    p.ws = ws
    return p


# --- enabled ---


def test_group_enabled_when_any_parent_enabled() -> None:
    group = ParentInfraGroup([_make_parent("http://a.url"), _make_parent("http://b.url", enabled=False)])
    assert group.enabled is True


def test_group_disabled_when_no_parents() -> None:
    assert ParentInfraGroup([]).enabled is False


def test_group_disabled_when_all_parents_disabled() -> None:
    group = ParentInfraGroup([_make_parent("http://a.url", enabled=False)])
    assert group.enabled is False


# --- ancestors ---


def test_ancestors_merges_parent_urls_and_their_chains() -> None:
    p1 = _make_parent("http://a.url", ancestors=["http://root.url"])
    p2 = _make_parent("http://b.url", ancestors=["http://root.url", "http://super.url"])
    group = ParentInfraGroup([p1, p2])
    result = group.ancestors
    assert [a.url for a in result] == ["http://a.url", "http://root.url", "http://b.url", "http://super.url"]


def test_ancestors_deduplicates() -> None:
    p1 = _make_parent("http://a.url", ancestors=["http://root.url"])
    p2 = _make_parent("http://a.url", ancestors=["http://root.url"])
    group = ParentInfraGroup([p1, p2])
    assert [a.url for a in group.ancestors] == ["http://a.url", "http://root.url"]


def test_ancestors_skips_disabled_parents() -> None:
    p1 = _make_parent("http://a.url", enabled=False, ancestors=["http://root.url"])
    p2 = _make_parent("http://b.url", ancestors=[])
    group = ParentInfraGroup([p1, p2])
    assert [a.url for a in group.ancestors] == ["http://b.url"]


def test_ancestors_empty_when_no_parents() -> None:
    assert ParentInfraGroup([]).ancestors == []


# --- send_topology_update ---


def test_send_topology_update_broadcasts_to_all_connected() -> None:
    p1 = _make_parent("http://a.url", ws=MagicMock())
    p2 = _make_parent("http://b.url", ws=MagicMock())
    group = ParentInfraGroup([p1, p2])
    group.send_topology_update("join", "http://child.url", "child", [])
    p1.task_manager.add_task_safe.assert_called_once()
    p2.task_manager.add_task_safe.assert_called_once()


def test_send_topology_update_skips_disconnected_parents() -> None:
    p1 = _make_parent("http://a.url", ws=None)
    p2 = _make_parent("http://b.url", ws=MagicMock())
    group = ParentInfraGroup([p1, p2])
    group.send_topology_update("join", "http://child.url", "child", [])
    p1.task_manager.add_task_safe.assert_not_called()
    p2.task_manager.add_task_safe.assert_called_once()


# --- send_models_list / send_usage ---


def test_send_models_list_broadcasts_to_all() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    group.send_models_list()
    p1.send_models_list.assert_called_once()
    p2.send_models_list.assert_called_once()


def test_send_usage_broadcasts_to_all() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    usage = MagicMock()
    group.send_usage(usage)
    p1.send_usage.assert_called_once_with(usage)
    p2.send_usage.assert_called_once_with(usage)


# --- check_subinfra_connection ---


def test_check_subinfra_connection_returns_true_if_any_matches() -> None:
    p1 = _make_parent("http://a.url")
    p1.check_subinfra_connection.return_value = False
    p2 = _make_parent("http://b.url")
    p2.check_subinfra_connection.return_value = True
    group = ParentInfraGroup([p1, p2])
    assert group.check_subinfra_connection(MagicMock()) is True


def test_check_subinfra_connection_returns_false_if_none_match() -> None:
    p1 = _make_parent("http://a.url")
    p1.check_subinfra_connection.return_value = False
    group = ParentInfraGroup([p1])
    assert group.check_subinfra_connection(MagicMock()) is False


# --- endpoint_registry setter ---


def test_endpoint_registry_setter_propagates_to_all_parents() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    registry = MagicMock()
    group.endpoint_registry = registry
    assert p1.endpoint_registry == registry
    assert p2.endpoint_registry == registry


# --- get_children setter ---


def test_get_children_setter_propagates_to_all_parents() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    group = ParentInfraGroup([p1, p2])
    callback = MagicMock()
    group.get_children = callback
    assert p1.get_children is callback
    assert p2.get_children is callback


def test_get_children_getter_defaults_to_dict() -> None:
    group = ParentInfraGroup([])
    assert group.get_children() == {}


# --- run ---


@pytest.mark.asyncio
async def test_run_empty_group_returns_immediately() -> None:
    group = ParentInfraGroup([])
    await group.run()  # should not raise


@pytest.mark.asyncio
async def test_run_with_parents_runs_all() -> None:
    p1 = _make_parent("http://a.url")
    p2 = _make_parent("http://b.url")
    p1.run = AsyncMock()
    p2.run = AsyncMock()
    group = ParentInfraGroup([p1, p2])
    await group.run()
    p1.run.assert_awaited_once()
    p2.run.assert_awaited_once()


# --- endpoint_registry getter ---


def test_endpoint_registry_getter_returns_none_when_no_parents() -> None:
    group = ParentInfraGroup([])
    assert group.endpoint_registry is None


# --- reconfigure ---


@pytest.mark.asyncio
async def test_reconfigure_replaces_parents_when_mesh_url_set() -> None:
    old_parent = _make_parent("http://old.url")
    old_parent.stop = AsyncMock()
    group = ParentInfraGroup([old_parent])
    registry = MagicMock()
    group.endpoint_registry = registry

    config = MagicMock()
    config.connect_to_mesh_url = "ws://new.url"
    task_manager = MagicMock()

    with patch("server.websockets.parent_infra_group.ParentInfra") as mock_parent_cls:
        new_parent = mock_parent_cls.return_value
        await group.reconfigure(config, task_manager)

    old_parent.stop.assert_awaited_once()
    mock_parent_cls.assert_called_once_with(config, task_manager, "ws://new.url")
    assert group.parents == [new_parent]
    assert new_parent.endpoint_registry == registry
    task_manager.add_task_safe.assert_called_once_with(new_parent.run.return_value, "parent_infra_group.reconfigure")


@pytest.mark.asyncio
async def test_reconfigure_clears_parents_when_mesh_url_empty() -> None:
    old_parent = _make_parent("http://old.url")
    old_parent.stop = AsyncMock()
    group = ParentInfraGroup([old_parent])

    config = MagicMock()
    config.connect_to_mesh_url = ""
    task_manager = MagicMock()

    await group.reconfigure(config, task_manager)

    old_parent.stop.assert_awaited_once()
    assert group.parents == []
    task_manager.add_task_safe.assert_not_called()


@pytest.mark.asyncio
async def test_reconfigure_skips_endpoint_registry_propagation_when_never_assigned() -> None:
    group = ParentInfraGroup([])

    config = MagicMock()
    config.connect_to_mesh_url = "ws://new.url"
    task_manager = MagicMock()

    with patch("server.websockets.parent_infra_group.ParentInfra") as mock_parent_cls:
        new_parent = mock_parent_cls.return_value
        await group.reconfigure(config, task_manager)

    assert group.parents == [new_parent]
    task_manager.add_task_safe.assert_called_once_with(new_parent.run.return_value, "parent_infra_group.reconfigure")


@pytest.mark.asyncio
async def test_reconfigure_propagates_endpoint_registry_when_group_started_with_no_parents() -> None:
    """Regression test: group started with an empty parents list (no `connect_to_mesh_url` at boot),
    so `endpoint_registry` was assigned via the setter while `self.parents` was still empty. The new
    parent created by a later `reconfigure()` call must still receive that registry."""
    group = ParentInfraGroup([])
    registry = MagicMock()
    group.endpoint_registry = registry

    config = MagicMock()
    config.connect_to_mesh_url = "ws://new.url"
    task_manager = MagicMock()

    with patch("server.websockets.parent_infra_group.ParentInfra") as mock_parent_cls:
        new_parent = mock_parent_cls.return_value
        await group.reconfigure(config, task_manager)

    assert group.parents == [new_parent]
    assert new_parent.endpoint_registry == registry


@pytest.mark.asyncio
async def test_reconfigure_propagates_get_children_to_new_parent() -> None:
    """Regression test: get_children must survive a live mesh reconfigure, not just process restart —
    otherwise the new parent silently reports an empty child list to its own parent up the mesh tree."""
    old_parent = _make_parent("http://old.url")
    old_parent.stop = AsyncMock()
    group = ParentInfraGroup([old_parent])
    callback = MagicMock()
    group.get_children = callback

    config = MagicMock()
    config.connect_to_mesh_url = "ws://new.url"
    task_manager = MagicMock()

    with patch("server.websockets.parent_infra_group.ParentInfra") as mock_parent_cls:
        new_parent = mock_parent_cls.return_value
        await group.reconfigure(config, task_manager)

    assert new_parent.get_children is callback


@pytest.mark.asyncio
async def test_reconfigure_defaults_get_children_when_never_assigned() -> None:
    group = ParentInfraGroup([])

    config = MagicMock()
    config.connect_to_mesh_url = "ws://new.url"
    task_manager = MagicMock()

    with patch("server.websockets.parent_infra_group.ParentInfra") as mock_parent_cls:
        new_parent = mock_parent_cls.return_value
        await group.reconfigure(config, task_manager)

    assert new_parent.get_children is dict
