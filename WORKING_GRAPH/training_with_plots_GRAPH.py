from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(project / "train.py"), "--family", "graph"], cwd=project, check=True)
