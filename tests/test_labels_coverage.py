from probe2circuit.coverage import chemical_coverage, label_support
from probe2circuit.labels import canonical, structural_labels


def fg(smi):
    lab = structural_labels(smi)
    return {k[3:] for k, v in lab.items() if k.startswith("fg_") and v}


def test_amino_acid_groups(analytes):
    by = {a["name"]: a["smiles"] for a in analytes}
    assert {"indole", "carboxylic_acid", "primary_amine"} <= fg(by["L-tryptophan"])
    assert "phenol" in fg(by["L-tyrosine"])
    assert "thiol" in fg(by["L-cysteine"])
    assert "thioether" in fg(by["L-methionine"])
    assert "imidazole_ring" in fg(by["L-histidine"])
    assert "guanidine" in fg(by["L-arginine"])
    assert "secondary_amine" in fg(by["L-proline"]) and "primary_amine" not in fg(by["L-proline"])
    assert "amide" in fg(by["L-asparagine"]) and "primary_amine" in fg(by["L-asparagine"])
    assert "phenol" not in fg(by["L-phenylalanine"]) and "benzene_ring" in fg(by["L-phenylalanine"])


def test_qm9_domain_rules(analytes):
    by = {a["name"]: structural_labels(a["smiles"]) for a in analytes}
    assert by["L-valine"]["in_qm9_domain"] == 1
    assert by["L-tryptophan"]["n_heavy"] == 15 and by["L-tryptophan"]["in_qm9_domain"] == 0
    assert by["L-cysteine"]["in_qm9_domain"] == 0          # sulfur
    assert by["indole"]["in_qm9_domain"] == 1
    assert by["indole-d6"]["in_qm9_domain"] == 0           # isotopologue
    assert canonical(next(a["smiles"] for a in analytes if a["name"] == "indole-d6")) == canonical("c1ccc2[nH]ccc2c1")


def test_stereo_ignored_for_matching():
    assert canonical("C[C@H](N)C(=O)O") == canonical("CC(N)C(=O)O")


def test_coverage_tables(analytes):
    qm9 = ["CC(N)C(=O)O", "NCC(=O)O", "c1ccccc1", "CCO"]
    chem = {r["analyte"]: r for r in chemical_coverage(analytes, qm9)}
    assert chem["L-alanine"]["exact_in_qm9"] == 1 and chem["L-alanine"]["nn_tanimoto"] == 1.0
    assert chem["L-tryptophan"]["exact_in_qm9"] == 0
    labs = {r["label"]: r for r in label_support(analytes, qm9)}
    assert labs["thiol"]["status"] == "NO QM9 SUPPORT"
    assert labs["alkyne"]["status"] == "unused"      # no analyte has one (isatin now has a ketone)
