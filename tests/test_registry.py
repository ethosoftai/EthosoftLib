import sys
import pytest
from ethosoftlib import ProviderRegistry, providers
from ethosoftlib.core.errors import ProviderError, ProviderNotFoundError, ProviderLoadError


def test_builtin_provider_is_lazy():
    # Generic SDK import and discovery must not import Mercan native bindings.
    assert providers.describe("inference", "mercan").name == "mercan"
    assert "ethosoftlib.mercan.native" not in sys.modules


def test_categories_are_arbitrary():
    registry = ProviderRegistry()
    registry.register("education", "sample", "builtins:list", description="Any domain")
    registry.register("storage", "sample", "builtins:dict")
    assert [p.category for p in registry.list()] == ["education", "storage"]
    assert registry.create("education", "sample", [1, 2]) == [1, 2]
    assert registry.create("storage", "sample", hello=3) == {"hello": 3}


def test_duplicates_and_unknown_provider():
    registry = ProviderRegistry()
    registry.register("tools", "demo", "builtins:list")
    with pytest.raises(ProviderError):
        registry.register("tools", "demo", "builtins:list")
    with pytest.raises(ProviderNotFoundError):
        registry.resolve("tools", "missing")


def test_bad_factory_is_rejected():
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.register("tools", "bad", "not-a-factory")
    registry.register("tools", "bad", "builtins:does_not_exist")
    with pytest.raises(ProviderLoadError):
        registry.resolve("tools", "bad")


def test_replace():
    registry = ProviderRegistry()
    registry.register("service", "x", "builtins:list")
    registry.register("service", "x", "builtins:dict", replace=True)
    assert registry.create("service", "x", a=1) == {"a": 1}
