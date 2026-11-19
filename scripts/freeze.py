"""Freeze a phase's settings before its one-time test run, or check the freeze.

Usage: python scripts/freeze.py --phase phase1            (write the manifest)
       python scripts/freeze.py --phase phase1 --check    (exit 1 if anything changed)

The logic lives in herdsight.utils.freeze; see its docstring.
"""

import sys

from herdsight.utils.freeze import main

if __name__ == "__main__":
    sys.exit(main())
