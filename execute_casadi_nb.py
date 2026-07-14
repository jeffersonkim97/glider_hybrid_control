"""Execute p1b/p1b_casadi_symbolic.ipynb inplace using nbclient with UTF-8 encoding."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import nbformat
from nbclient import NotebookClient

nb_path = os.path.join(os.path.dirname(__file__), 'p1b', 'p1b_casadi_symbolic.ipynb')

with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name='python3',
    resources={'metadata': {'path': os.path.join(os.path.dirname(__file__), 'p1b')}},
)
client.execute()

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Done. Executed {len(nb.cells)} cells.')
