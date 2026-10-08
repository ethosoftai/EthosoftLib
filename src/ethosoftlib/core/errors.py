"""Errors shared by all integrations, without a dependency on any provider."""

class EthosoftError(Exception):
    """Base SDK error."""


class ProviderError(EthosoftError):
    """Provider registration or construction failed."""


class ProviderNotFoundError(ProviderError, LookupError):
    """No provider matches the requested category/name."""


class ProviderLoadError(ProviderError):
    """The registered provider could not be imported or instantiated."""
