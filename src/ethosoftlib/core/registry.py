"""Generic lazy provider registry (not tied to inference or any model API)."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable
import re

from .errors import ProviderError, ProviderLoadError, ProviderNotFoundError

_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata for an integration; `factory` is resolved only when requested."""

    category: str
    name: str
    factory: str
    description: str = ""


class ProviderRegistry:
    """Registry of independent providers, grouped by arbitrary categories.

    Third-party packages may use categories such as 'inference', 'storage',
    'education', 'analytics', or their own domain. No dependencies are imported
    during registration or listing.
    """

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ProviderInfo] = {}

    def register(
        self, category: str, name: str, factory: str, *,
        description: str = "", replace: bool = False,
    ) -> ProviderInfo:
        if not _NAME.fullmatch(category) or not _NAME.fullmatch(name):
            raise ValueError("category/name must be lowercase identifiers")
        if ":" not in factory or not all(factory.split(":", 1)):
            raise ValueError("factory must be 'python.module:CallableName'")
        info = ProviderInfo(category, name, factory, description)
        key = (category, name)
        if key in self._items and not replace:
            raise ProviderError(f"Provider already registered: {category}/{name}")
        self._items[key] = info
        return info

    def list(self, category: str | None = None) -> tuple[ProviderInfo, ...]:
        """Discover registered providers without importing them."""
        return tuple(
            value for key, value in sorted(self._items.items())
            if category is None or key[0] == category
        )

    def describe(self, category: str, name: str) -> ProviderInfo:
        try:
            return self._items[(category, name)]
        except KeyError as exc:
            raise ProviderNotFoundError(f"Unknown provider: {category}/{name}") from exc

    def resolve(self, category: str, name: str) -> Callable[..., Any]:
        """Import a provider's public constructor only when explicitly requested."""
        info = self.describe(category, name)
        module_name, attribute = info.factory.split(":", 1)
        try:
            module = import_module(module_name)
            target = module
            for member in attribute.split("."):
                target = getattr(target, member)
        except (ImportError, AttributeError) as exc:
            raise ProviderLoadError(
                f"Could not import {info.factory} for {category}/{name}: {exc}"
            ) from exc
        if not callable(target):
            raise ProviderLoadError(f"Provider factory is not callable: {info.factory}")
        return target

    def create(self, category: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create a provider instance using arbitrary provider-specific arguments."""
        return self.resolve(category, name)(*args, **kwargs)


providers = ProviderRegistry()
# Optional integration: registering it does NOT import it or load libmercan.
providers.register(
    "inference", "mercan", "ethosoftlib.mercan:MercanRuntime",
    description="MercanRuntime native inference via the stable C ABI",
)

# This tokenizer is a separate provider: native code is never loaded on discovery.
providers.register(
    "tokenizer", "nedo", "ethosoftlib.nedo:Tokenizer",
    description="Original Rust NDSRF004 tokenizer, no LLM required",
)
