"""Three small helpers shared by the demo notebooks.

- `nb.bootstrap()` finds the repo root, loads `.env`, returns the model config.
- `ui.table(...)` and `ui.diagram(...)` render a list of dicts as a table and
  embed one of the SVGs in `notebooks/data/diagrams/`.

Trimmed copies of the same modules in the AI Forward Deployed Engineer
Certification repo, so a notebook here reads the same way as one there.
"""

from .config import Config
from .nb import bootstrap, find_root
from .ui import diagram, lines, spinner, table, track

__all__ = ["Config", "bootstrap", "find_root", "diagram", "lines", "spinner", "table", "track"]
