"""Optional adapter for MercanRuntime; no native code is loaded on import."""
from .runtime import MercanRuntime, MercanModel, MercanContext, MercanEmbedder
from .model import Model, ChatSession
from ethosoftlib.nedo import NedoTokenizer
from .native import MercanError, MercanLibraryNotFound

__all__ = [
    "MercanRuntime", "MercanModel", "MercanContext", "MercanEmbedder", "Model", "ChatSession", "NedoTokenizer",
    "MercanError", "MercanLibraryNotFound",
]
