import sys
from pathlib import Path

# make the repository-root modules (linearity, theory, ...) importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
