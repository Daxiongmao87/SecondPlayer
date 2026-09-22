import pytest

from secondplayer.adapters.loader import load_adapter_module
from secondplayer.errors import AdapterError


def test_missing_adapter_has_clean_error():
    with pytest.raises(AdapterError, match="not installed"):
        load_adapter_module("does-not-exist")
