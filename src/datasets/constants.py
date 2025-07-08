# datasets/constants.py
"""
Constants used throughout the molecular datasets.
"""

from rdkit.Chem.rdchem import HybridizationType

# Atom feature constants
ATOM_TYPES = list(range(1, 119))  # Atomic numbers from 1 to 118
DEGREES = list(range(6))          # Degrees from 0 to 5
HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2
]