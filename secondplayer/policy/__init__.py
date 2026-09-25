from __future__ import annotations

from __future__ import annotations

from ..config import RuntimeConfig
from .laya import LayaVisionPolicy
from .openjev import OPENJEV_MODEL_IDS, OpenJevPolicy
from .playjev import PLAYJEV_MODEL_IDS, PlayJevPolicy


def create_policy(config: RuntimeConfig) -> LayaVisionPolicy | PlayJevPolicy | OpenJevPolicy:
    """Pick the vision backend by model id. All backends share the decide() interface."""
    model_id = config.model.strip().lower()
    if model_id in PLAYJEV_MODEL_IDS:
        return PlayJevPolicy(config)
    if model_id in OPENJEV_MODEL_IDS:
        return OpenJevPolicy(config)
    return LayaVisionPolicy(config)


__all__ = ["LayaVisionPolicy", "OpenJevPolicy", "PlayJevPolicy", "create_policy"]
