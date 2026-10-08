"""EthosoftLib: independent, optional integrations under one Python SDK.

Importing ethosoftlib never loads Mercan, native binaries, or third-party SDKs.
"""
from .core.registry import ProviderInfo, ProviderRegistry, providers

__version__ = "0.3.0"
__all__ = ["ProviderInfo", "ProviderRegistry", "providers", "__version__"]
