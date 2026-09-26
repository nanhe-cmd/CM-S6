import importlib
import sys
from pathlib import Path

from torch import nn


BASELINES = {
    "s2mamba": "S2-Mamba",
    "dsformer": "DSFormer",
    "mft": "MFT",
    "picnet": "PICNet",
    "mtmixer": "MTMixer",
    "hlmamba": "HLMamba",
    "mcamamba": "MCAMamba",
    "msfmamba": "MSFMamba",
}


def load_model(factory_reference, config, external_root="external"):
    root = Path(external_root).expanduser().resolve()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if ":" not in factory_reference:
        raise ValueError("Factory must use the form package.module:function")
    module_name, function_name = factory_reference.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Could not import {module_name}. Check --external-root and the adapter module."
        ) from error
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError(f"Factory is not callable: {factory_reference}")

    model = factory(config)
    if not isinstance(model, nn.Module):
        raise TypeError("The external factory must return torch.nn.Module")
    return model
