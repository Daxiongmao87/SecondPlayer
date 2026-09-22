class SecondPlayerError(RuntimeError):
    """Base error surfaced to the CLI."""


class ConfigurationError(SecondPlayerError):
    pass


class AdapterError(SecondPlayerError):
    pass


class PlatformError(SecondPlayerError):
    pass


class ModelError(SecondPlayerError):
    pass
