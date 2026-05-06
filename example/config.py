from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.py"
_SPEC = spec_from_file_location("project_config", _ROOT_CONFIG_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.isupper():
        globals()[_name] = getattr(_MODULE, _name)
