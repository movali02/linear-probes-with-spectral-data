"""Structural labels from SMILES with RDKit.

These are the probe targets that exist on BOTH sides (QM9S and SERS), unlike
analyte identity, which only exists for SERS. Labels are computed from the
neutral SMILES; in water the amino acids are zwitterions, which changes the
carboxylate / ammonium bands but not which groups are present."""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# name -> SMARTS. Kept deliberately small and chemically unambiguous.
SMARTS = {
    "carboxylic_acid":    "[CX3](=O)[OX2H1,OX1-]",
    "primary_amine":      "[NX3;H2;!$(NC=[O,S,N]);!$(N-a)]",
    "secondary_amine":    "[NX3;H1;!$(NC=[O,S,N]);!$(N-a);!a]([#6])[#6]",
    "amide":              "[NX3][CX3](=[OX1])[#6]",
    "guanidine":          "[NX3,NX2][CX3](=[NX2,NX3+])[NX3]",
    "hydroxyl_aliphatic": "[OX2H][CX4]",
    "phenol":             "[OX2H]c",
    "thiol":              "[SX2H]",
    "thioether":          "[SX2]([#6])[#6]",
    "ketone":             "[#6][CX3](=O)[#6]",
    "aldehyde":           "[CX3H1](=O)[#1,#6]",
    "ester":              "[#6][CX3](=O)[OX2][#6]",
    "ether":              "[OD2;!$(OC=O)]([#6;!$(C=O)])[#6;!$(C=O)]",
    "nitrile":            "[CX2]#[NX1]",
    "alkyne":             "[CX2]#[CX2]",
    "alkene":             "[CX3;!a]=[CX3;!a]",
    "fluoro":             "[F]",
    "aromatic_ring":      "a",
    "benzene_ring":       "c1ccccc1",
    "pyrrole_ring":       "[nX3H1,nX3]1cccc1",
    "imidazole_ring":     "c1cnc[nH]1",
    "indole":             "c1ccc2c(c1)cc[nX3]2",
    "lactam":             "[NX3;R][CX3;R](=O)",
}
_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
assert all(p is not None for p in _PATTERNS.values()), "bad SMARTS"

QM9_ELEMENTS = {"C", "N", "O", "F"}
QM9_MAX_HEAVY = 9


@lru_cache(maxsize=None)
def mol_from_smiles(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"RDKit cannot parse SMILES {smi!r}")
    return m


def strip_isotopes(m):
    """Drop isotope labels and the explicit [2H] atoms they create."""
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        a.SetIsotope(0)
    return Chem.RemoveHs(m)


def canonical(smi: str, isomeric: bool = False) -> str:
    """Non-isomeric by default: QM9 SMILES and your L-amino-acid SMILES differ in
    stereo annotation, and Raman doesn't resolve enantiomers anyway. Isotopes are
    also dropped (indole-d6 -> indole) so matching is by constitution. Note that
    deuteration DOES shift Raman bands, so the spectra are not interchangeable."""
    return Chem.MolToSmiles(strip_isotopes(mol_from_smiles(smi)), isomericSmiles=isomeric)


def structural_labels(smi: str) -> dict:
    m0 = mol_from_smiles(smi)
    has_iso = any(a.GetIsotope() for a in m0.GetAtoms())
    m = strip_isotopes(m0)
    elems = sorted({a.GetSymbol() for a in m.GetAtoms()})
    n_heavy = m.GetNumHeavyAtoms()
    out = {f"fg_{k}": int(m.HasSubstructMatch(p)) for k, p in _PATTERNS.items()}
    out.update({
        "n_heavy": n_heavy,
        "elements": "".join(elems),
        "n_rings": rdMolDescriptors.CalcNumRings(m),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
        "has_S": int("S" in elems),
        "has_isotope": int(has_iso),
        # QM9 has no isotopologues, so a deuterated analyte is out of domain spectrally
        "in_qm9_domain": int(n_heavy <= QM9_MAX_HEAVY and set(elems) <= QM9_ELEMENTS and not has_iso),
    })
    return out


FG_KEYS = [f"fg_{k}" for k in SMARTS]


def add_labels(records: list[dict]) -> int:
    """Attach labels in place to every record with a SMILES. Returns #unlabelled."""
    missing = 0
    for r in records:
        if r.get("smiles"):
            r["labels"] = structural_labels(r["smiles"])
        else:
            missing += 1
    return missing
