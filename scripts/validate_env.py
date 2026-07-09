"""Environment validation script (FIAP Tech Challenge, Stage 2).

Verifies that a freshly cloned environment satisfies every project
requirement: Python version, required libraries, environment template
and reproducibility helpers. Exits non-zero on any failure so it can
gate CI pipelines.

Usage:
    uv run python scripts/validate_env.py
"""

import importlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("validate_env")

REQUIRED_PYTHON = (3, 11)
REQUIRED_PACKAGES = ("torch", "sklearn", "mlflow", "dvc", "pandas", "numpy", "pydantic_settings")


def check_python_version() -> bool:
    """Ensure the interpreter matches the version pinned in pyproject.

    Returns:
        True when the running Python is at least :data:`REQUIRED_PYTHON`.
    """
    current = sys.version_info[:2]
    ok = current >= REQUIRED_PYTHON
    status = "OK" if ok else "FAIL"
    logger.info("[%s] Python %d.%d (required >= %d.%d)", status, *current, *REQUIRED_PYTHON)
    return ok


def check_packages() -> bool:
    """Import every required package and report its version.

    Returns:
        True when all packages in :data:`REQUIRED_PACKAGES` import.
    """
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            logger.info("[OK] %s %s", name, version)
        except ImportError:
            logger.error("[FAIL] %s not importable — run `uv sync`", name)
            all_ok = False
    return all_ok


def check_env_file() -> bool:
    """Verify ``.env`` exists and covers every key in ``.env.example``.

    Returns:
        True when ``.env`` is present and no template key is missing.
    """
    example, env = Path(".env.example"), Path(".env")
    if not env.exists():
        logger.warning("[WARN] .env not found — copy .env.example: `cp .env.example .env`")
        return False
    expected = _env_keys(example)
    missing = expected - _env_keys(env)
    if missing:
        logger.error("[FAIL] .env is missing keys: %s", sorted(missing))
        return False
    logger.info("[OK] .env present with all %d expected keys", len(expected))
    return True


def _env_keys(path: Path) -> set[str]:
    """Extract variable names from a dotenv-style file.

    Args:
        path: File to parse; missing files yield an empty set.

    Returns:
        The set of variable names declared in the file.
    """
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = [line for line in lines if "=" in line and not line.lstrip().startswith("#")]
    return {line.split("=", 1)[0].strip() for line in entries}


def check_reproducibility() -> bool:
    """Confirm settings load and the global seed can be applied.

    Returns:
        True when :mod:`recsys.config` and :mod:`recsys.seeding` work.
    """
    try:
        from recsys.config import get_settings
        from recsys.seeding import set_global_seed
    except ImportError as exc:
        logger.error("[FAIL] recsys package not importable: %s", exc)
        return False
    settings = get_settings()
    set_global_seed(settings.random_seed)
    logger.info("[OK] settings loaded, global seed=%d applied", settings.random_seed)
    return True


def main() -> int:
    """Run every check and return a shell-friendly exit code.

    Returns:
        ``0`` when all checks pass, ``1`` otherwise.
    """
    checks = (check_python_version(), check_packages(), check_env_file(), check_reproducibility())
    if all(checks):
        logger.info("Environment OK — ready to run the pipeline.")
        return 0
    logger.error("Environment validation FAILED — fix the items above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
