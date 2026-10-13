import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "training"), os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
