"""Sphinx configuration file."""

from __future__ import annotations

import warnings
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import pybtex.plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import field, href

ROOT = Path(__file__).parent.parent.resolve()


try:
    version = metadata.version("qtpu")
except ModuleNotFoundError:
    msg = (
        "Package should be installed to produce documentation! "
        "Assuming a modern git archive was used for version discovery."
    )
    warnings.warn(msg, stacklevel=1)

    from setuptools_scm import get_version

    version = get_version(root=str(ROOT), fallback_root=ROOT)

# Filter git details from version
release = version.split("+")[0]
if TYPE_CHECKING:
    from pybtex.database import Entry
    from pybtex.richtext import HRef

project = "qTPU"
author = "Nathaniel Tornow, Leibniz Supercomputing Centre"
language = "en"
project_copyright = "TODO"
# -- General configuration ---------------------------------------------------

master_doc = "index"

templates_path = ["_templates"]
html_css_files = ["custom.css"]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinxext.opengraph",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]

pygments_style = "colorful"

add_module_names = False

modindex_common_prefix = ["qtpu."]

# Update intersphinx mapping to include specific Qiskit classes
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    # "qiskit.circuit": ("https://qiskit.org/documentation/stubs/qiskit.circuit.html", None),
}

nbsphinx_execute = "always"
nbsphinx_allow_errors = False
highlight_language = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=200",
]
nbsphinx_kernel_name = "python3"

autosectionlabel_prefix_document = True

exclude_patterns = ["_build", "build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env"]

# Add autoapi configuration
autoapi_type = "python"
autoapi_dirs = ["../src/qtpu"]
autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "show-inheritance-diagram",
]

# Add nitpick_ignore for troublesome types
nitpick_ignore = [
    ("py:class", "QuantumCircuit"),
    ("py:class", "optional"),
    ("py:class", "qiskit.circuit.quantumcircuit.QuantumCircuit"),
]

# Add type aliases for documentation
autodoc_type_aliases = {
    'QuantumCircuit': 'qiskit.circuit.QuantumCircuit',
}

class CDAStyle(UnsrtStyle):
    """Custom style for including PDF links."""

    def format_url(self, _e: Entry) -> HRef:
        """Format URL field as a link to the PDF."""
        url = field("url", raw=True)
        return href()[url, "[PDF]"]


pybtex.plugin.register_plugin("pybtex.style.formatting", "cda_style", CDAStyle)

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "cda_style"

copybutton_prompt_text = r"(?:\(venv\) )?(?:\[.*\] )?\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

autosummary_generate = True


typehints_use_rtype = False
napoleon_use_rtype = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/nathanieltornow/qtpu",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}
