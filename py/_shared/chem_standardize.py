"""Shared molecule-standardization logic.

Deliberately the ONE exception to this codebase's usual per-track
(Classification/Regression) and per-node code duplication convention: this
specific chemistry must be bit-identical between whatever standardizes
training compounds (01 Data Load & Standardization, both tracks) and
whatever standardizes compounds at inference/screening time (the
Screener) -- if the two paths diverge, the same physical molecule can be
represented differently (or dropped entirely by one path and not the
other) depending on which stage sees it, silently breaking the model's
train/inference consistency. Everything else in this codebase should stay
duplicated per the established pattern; only this module is shared.
"""
from typing import Any, Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

METAL_IONS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
}

_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_mol(mol: Optional[Chem.Mol]) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
    """
    Standardizes a single molecule: rejects unparseable/metal-only inputs,
    then applies Cleanup -> FragmentParent (largest-fragment retention,
    replaces salts/counter-ions -- e.g. "CCO.[Na+]" keeps "CCO", it is NOT
    rejected outright for having more than one fragment) -> Uncharger
    (protonation-state normalization) -> TautomerEnumerator.Canonicalize
    (canonical tautomer). Returns (standardized_mol_or_None, info) where
    info records what changed.
    """
    info = {"status": "ok", "reason": "", "fragment_changed": False,
            "charge_changed": False, "tautomer_changed": False}
    if mol is None:
        info["status"], info["reason"] = "rejected", "unparseable"
        return None, info
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atom_symbols and atom_symbols.issubset(METAL_IONS):
        info["status"], info["reason"] = "rejected", "metal_only"
        return None, info
    try:
        smi0 = Chem.MolToSmiles(mol)
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        smi1 = Chem.MolToSmiles(mol)
        info["fragment_changed"] = (smi1 != smi0)
        mol = _UNCHARGER.uncharge(mol)
        smi2 = Chem.MolToSmiles(mol)
        info["charge_changed"] = (smi2 != smi1)
        mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
        smi3 = Chem.MolToSmiles(mol)
        info["tautomer_changed"] = (smi3 != smi2)
        if mol is None or mol.GetNumAtoms() == 0:
            info["status"], info["reason"] = "rejected", "empty_after_standardization"
            return None, info
        return mol, info
    except Exception as e:
        info["status"], info["reason"] = "rejected", f"standardization_error: {e}"
        return None, info
