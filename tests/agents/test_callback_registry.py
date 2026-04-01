"""Tests for the callback registry.

Validates registration, lookup, duplicate detection, and introspection
of the callback registry used by agent pipelines.
"""

from __future__ import annotations

import pytest

from healthbench_agent.agent.callback_registry import (
    _REGISTRY,
    get_callback,
    register_callback,
    registered_callbacks,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear callback registry before and after each test."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


class TestRegisterCallback:
    """Tests for the @register_callback decorator."""

    def test_registers_function(self):
        @register_callback("my_callback")
        def my_callback():
            pass

        assert "my_callback" in _REGISTRY
        assert _REGISTRY["my_callback"] is my_callback

    def test_returns_original_function(self):
        @register_callback("original")
        def original():
            pass

        assert original.__name__ == "original"

    def test_duplicate_raises_value_error(self):
        @register_callback("dup")
        def first():
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_callback("dup")
            def second():
                pass


class TestGetCallback:
    """Tests for the get_callback lookup function."""

    def test_returns_registered_callback(self):
        @register_callback("found")
        def found():
            pass

        assert get_callback("found") is found

    def test_unregistered_raises_key_error(self):
        with pytest.raises(KeyError, match="not registered"):
            get_callback("nonexistent")

    def test_error_message_lists_available(self):
        @register_callback("available_one")
        def one():
            pass

        with pytest.raises(KeyError, match="available_one"):
            get_callback("missing")


class TestRegisteredCallbacks:
    """Tests for the registered_callbacks introspection function."""

    def test_returns_copy(self):
        @register_callback("copy_test")
        def copy_test():
            pass

        result = registered_callbacks()
        assert "copy_test" in result
        # Mutating the copy does not affect the registry.
        result.pop("copy_test")
        assert "copy_test" in _REGISTRY

    def test_empty_when_no_callbacks(self):
        assert registered_callbacks() == {}
