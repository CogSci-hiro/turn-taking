from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path


# -- Path setup --------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Ensure Matplotlib is usable in headless doc builds and has a writable cache.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


# -- Project information -----------------------------------------------------
def _load_pyproject() -> dict:
    try:
        import tomllib  # py>=3.11
    except Exception:  # pragma: no cover
        return {}
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return {}
    try:
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return {}


_pp = _load_pyproject()
_proj = _pp.get("project", {}) if isinstance(_pp, dict) else {}

project = str(_proj.get("name", "turntaking"))
author = str(_proj.get("author", "turntaking contributors"))
copyright = f"2026, {author}"

try:
    import importlib.metadata as _metadata

    release = _metadata.version(project)
except Exception:  # noqa: BLE001
    release = str(_proj.get("version", "0.0.0"))

# Sphinx uses `version` (short X.Y) and `release` (full) in a few places
# including the objects inventory header.
version = ".".join(str(release).split(".")[:2]) if str(release) else "0.0"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autosummary_generate = True
autosummary_imported_members = True

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Mock optional/extra-heavy dependencies that are not needed to render API docs.
autodoc_mock_imports = [
    "cairosvg",
    "PIL",
    "pyarrow",
]

default_role = "py:obj"
todo_include_todos = False

# MNE-style docstrings are typically NumPyDoc.
numpydoc_show_class_members = False

try:
    from intersphinx_registry import get_intersphinx_mapping

    intersphinx_mapping = get_intersphinx_mapping(
        packages={"python", "numpy", "scipy", "matplotlib"}
    )
except Exception:  # noqa: BLE001
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", {}),
        "numpy": ("https://numpy.org/doc/stable", {}),
        "scipy": ("https://docs.scipy.org/doc/scipy/", {}),
        "matplotlib": ("https://matplotlib.org/stable", {}),
    }


# -- Bibliography ------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]


# -- Sphinx-gallery ----------------------------------------------------------
sphinx_gallery_conf = {
    # As requested: source examples and generated gallery live in the same dir.
    "examples_dirs": ["auto_examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_.*\\.py",
    # Do not treat existing `index.rst` files as user-provided. When the gallery
    # output dir is the same as the examples dir, Sphinx-Gallery may leave a
    # generated `index.rst` behind, and its default `copyfile_regex=""` matches
    # everything, which flips `sg_root_index` to False and makes the build scan
    # generated subfolders like `images/` as "subsections".
    "copyfile_regex": r"^$",
    # When gallery output is written into the same directory as the examples,
    # Sphinx-Gallery creates an `images/` subdirectory. If nested sections are
    # enabled, that generated folder is mistaken for a subsection and triggers
    # missing GALLERY_HEADER errors. Keep the gallery flat.
    "nested_sections": False,
    "download_all_examples": False,
    "within_subsection_order": "FileNameSortKey",
}


# -- Linkcode (optional) -----------------------------------------------------
def linkcode_resolve(domain: str, info: dict) -> str | None:
    """
    Resolve links to source code.

    This project does not assume a canonical remote URL. If you set the env var
    ``TURNTAKING_REPO_URL`` (e.g. a GitHub HTTPS URL), linkcode will attempt to
    construct stable links for Python objects.
    """
    if domain != "py":
        return None
    repo = os.environ.get("TURNTAKING_REPO_URL", "").rstrip("/")
    if not repo:
        return None

    modname = info.get("module", "")
    fullname = info.get("fullname", "")
    if not modname:
        return None

    try:
        module = __import__(modname, fromlist=["*"])
    except Exception:  # noqa: BLE001
        return None

    obj = module
    for part in fullname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        fn = inspect.getsourcefile(obj) or inspect.getsourcefile(module)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:  # noqa: BLE001
        return None

    if fn is None:
        return None

    rel = Path(fn).resolve().relative_to(ROOT).as_posix()
    end = lineno + len(source) - 1
    return f"{repo}/blob/main/{rel}#L{lineno}-L{end}"


# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "navbar_align": "content",
}
