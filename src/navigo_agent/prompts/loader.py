"""Prompt loader: YAML-based prompt management with variable interpolation.

Design:
  - Prompts live in YAML files under `src/navigo_agent/prompts/*.yaml`
  - Each file has a top-level key per prompt variant (e.g. `react_system`)
  - `render_prompt()` does `str.format(**kwargs)` for variable injection
  - Versioning: each prompt has an optional `version` field for change tracking
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent

# ── In-memory prompt cache (lazy-loaded) ───────────────────────────────

_cache: dict[str, dict] = {}


def _load_yaml_file(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_yaml_available():
    """Ensure PyYAML is importable — fail early with a clear message."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyYAML is required for prompt management. "
            "Install it with: uv add pyyaml"
        )


def load_prompt(name: str) -> str:
    """Load a prompt template by dotted name: '{file}.{key}'.

    Example:
        load_prompt("flight.react_system")  → loads flight.yaml, key 'react_system'
        load_prompt("supervisor.system")    → loads supervisor.yaml, key 'system'

    Prompts are cached in memory after first load.
    """
    _ensure_yaml_available()

    if "." not in name:
        raise ValueError(f"Prompt name must be 'file.key', got: {name}")

    file_name, key = name.split(".", 1)

    # Check cache
    if file_name not in _cache:
        yaml_path = PROMPT_DIR / f"{file_name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {yaml_path}")
        _cache[file_name] = _load_yaml_file(yaml_path)

    prompts = _cache[file_name]
    if key not in prompts:
        available = ", ".join(prompts.keys())
        raise KeyError(f"Prompt key '{key}' not found in {file_name}.yaml. Available: {available}")

    entry = prompts[key]
    if isinstance(entry, dict):
        return entry.get("template", "")
    return str(entry)


def render_prompt(template: str, **kwargs) -> str:
    """Render a prompt template with variable substitution.

    Uses str.format() — variables in the template are {variable_name}.

    Example:
        render_prompt("Hello {name}!", name="World")  → "Hello World!"
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning("Missing variable in prompt template: %s", e)
        # Return template with missing vars left in place (graceful degradation)
        return template
    except Exception as e:
        logger.error("Failed to render prompt template: %s", e)
        return template


def list_prompts() -> dict[str, list[str]]:
    """List all available prompt files and their keys."""
    _ensure_yaml_available()

    result: dict[str, list[str]] = {}
    for yaml_path in sorted(PROMPT_DIR.glob("*.yaml")):
        file_name = yaml_path.stem
        if file_name.startswith("_"):
            continue
        prompts = _load_yaml_file(yaml_path)
        result[file_name] = list(prompts.keys())

    return result
