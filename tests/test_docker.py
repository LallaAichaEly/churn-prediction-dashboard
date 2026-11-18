"""L'image Docker de l'API n'installe que requirements-api.txt : ces tests vérifient, sans Docker,
qu'il ne lui manque rien. (La CI `.github/workflows/docker.yml` construit et interroge la vraie image.)"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Le code qui entre dans l'image (voir .dockerignore et Dockerfile).
IMAGE_CODE = [ROOT / "api.py", ROOT / "scripts" / "train.py", *sorted((ROOT / "src" / "churn").glob("*.py"))]
LOCAL_PACKAGES = {"churn"}
# Nom à l'import → nom du paquet sur PyPI, quand ils diffèrent.
PYPI_NAME = {"sklearn": "scikit-learn"}
# Installés avec un autre paquet de la liste (pydantic et starlette viennent avec fastapi).
BUNDLED = {"pydantic": "fastapi", "starlette": "fastapi"}


def requirements(name: str) -> set[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return {re.split(r"[<>=!~\[; ]", line.strip(), maxsplit=1)[0].lower()
            for line in lines if line.strip() and not line.lstrip().startswith(("#", "-"))}


def third_party_imports(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules - set(sys.stdlib_module_names) - LOCAL_PACKAGES - {"__future__"}


def test_image_code_only_imports_what_the_image_installs():
    """Ajouter `import plotly` dans src/churn/model.py casserait l'image : ce test le voit tout de suite."""
    installed = requirements("requirements-api.txt")
    for path in IMAGE_CODE:
        for module in third_party_imports(path):
            package = BUNDLED.get(module, PYPI_NAME.get(module, module)).lower()
            assert package in installed, (
                f"{path.relative_to(ROOT)} importe « {module} » : ajoutez « {package} » à requirements-api.txt"
            )


def test_api_requirements_are_a_subset_of_the_full_requirements():
    """Une seule liste fait foi pour le développement : requirements.txt doit tout contenir."""
    missing = requirements("requirements-api.txt") - requirements("requirements.txt")
    assert not missing, f"à ajouter aussi dans requirements.txt : {sorted(missing)}"


def test_dockerignore_lets_the_image_code_in():
    """Le .dockerignore est une liste blanche : chaque fichier copié par le Dockerfile doit y être autorisé."""
    allowed = [line[1:] for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
               if line.startswith("!")]
    for path in [*IMAGE_CODE, ROOT / "requirements-api.txt"]:
        relative = path.relative_to(ROOT)
        assert any(relative.match(pattern) for pattern in allowed), f"{relative} est exclu par .dockerignore"
