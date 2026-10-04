"""DFT kit: geometry validation, structure checks on final geometries, ORCA
output parsing, sticks, and the gap-mode orientation average."""
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dft"))
import dft_check  # noqa: E402

L_THR = "C[C@@H](O)[C@H](N)C(=O)O"
ALLO_THR = "C[C@H](O)[C@H](N)C(=O)O"
D_THR = "C[C@H](O)[C@@H](N)C(=O)O"          # mirror image of L-Thr
GLY_ZW = "[NH3+]CC(=O)[O-]"
HIS_TAU = "N[C@@H](Cc1c[nH]cn1)C(=O)O"
HIS_PI = "N[C@@H](Cc1cnc[nH]1)C(=O)O"


def xyz_of(smiles, seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    c = m.GetConformer()
    rows = [f"{a.GetSymbol()} {c.GetAtomPosition(a.GetIdx()).x:.5f} {c.GetAtomPosition(a.GetIdx()).y:.5f} "
            f"{c.GetAtomPosition(a.GetIdx()).z:.5f}" for a in m.GetAtoms()]
    return "\n".join([str(len(rows)), "t"] + rows)


def test_same_structure_passes():
    r = dft_check.compare_structure(xyz_of(L_THR), L_THR)
    assert r["constitution_ok"] and r["stereo"] == "same"


def test_mirror_image_is_accepted():
    assert dft_check.compare_structure(xyz_of(D_THR), L_THR)["stereo"] == "same"


def test_allo_threonine_is_caught():
    r = dft_check.compare_structure(xyz_of(ALLO_THR), L_THR)
    assert r["constitution_ok"] and r["stereo"] == "diastereomer"


def test_zwitterion_that_lost_its_proton_is_caught():
    r = dft_check.compare_structure(xyz_of("NCC(=O)O"), GLY_ZW, charge=0)
    assert not r["constitution_ok"] and "H on N" in r["detail"]


def test_histidine_tautomer_switch_is_caught():
    assert not dft_check.compare_structure(xyz_of(HIS_PI), HIS_TAU)["constitution_ok"]


FAKE_OUT = """
* xyz 0 1
! CPCM(Water)
                     *******************
                     *    HURRAY       *
                     *******************
          ***********************************************
          *  FINAL ENERGY EVALUATION AT THE STATIONARY POINT  *
          ***********************************************
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
{coords}

-----------------------
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:         0.00 cm**-1
     6:       {f6} cm**-1{imag}
     7:      1003.20 cm**-1
     8:      1620.00 cm**-1

------------
NORMAL MODES
------------

--------------
RAMAN SPECTRUM
--------------

 Mode    freq (cm**-1)   Activity   Depolarization
-------------------------------------------------------------------
   6:      512.10      2.000000      0.750000
   7:     1003.20     30.000000      0.100000
   8:     1620.00     10.000000      0.500000

The first frequency considered to be a vibration is 6
                             ****ORCA TERMINATED NORMALLY****
"""


def write_job(tmp_path, smiles, f6="512.10", imag=""):
    coords = "\n".join("  " + l for l in xyz_of(smiles).splitlines()[2:])
    wd = tmp_path / "job"
    wd.mkdir(exist_ok=True)
    (wd / "job.out").write_text(FAKE_OUT.format(coords=coords, f6=f6, imag=imag))
    return wd


def test_check_job_pass(tmp_path):
    r = dft_check.check_job(write_job(tmp_path, GLY_ZW), "job", GLY_ZW, 0)
    assert r["verdict"] == "PASS", r
    assert r["n_raman_modes"] == 3 and r["solvent"] == "water" and r["n_imaginary"] == 0


def test_check_job_imaginary_and_proton_transfer(tmp_path):
    r = dft_check.check_job(write_job(tmp_path, "NCC(=O)O", f6="-45.20", imag=" ***imaginary mode***"),
                            "job", GLY_ZW, 0)
    assert r["verdict"] == "FAIL" and r["n_imaginary"] == 1 and "structure changed" in r["detail"]


# ------------------------------------------------------------ sticks + gap mode
import dft_to_sticks  # noqa: E402
import gapmode  # noqa: E402


def test_invariants_roundtrip():
    rng = np.random.default_rng(0)
    a2, g2 = rng.uniform(0, 2, 50), rng.uniform(0, 5, 50)
    S = 45 * a2 + 7 * g2
    rho = 3 * g2 / (45 * a2 + 4 * g2)
    a2b, g2b = dft_to_sticks.invariants(S, rho)
    assert np.allclose(a2, a2b) and np.allclose(g2, g2b)


def test_frequency_factor_size_at_785():
    f = dft_to_sticks.freq_factor(np.array([500.0, 1700.0]))
    assert 5.0 < f[0] / f[1] < 6.5          # ~5.6x: a 500 band vs a 1700 band of equal activity


def _fake_hess(path, tensors, order=("xx", "yy", "zz", "xy", "xz", "yz"), freqs=None):
    n_vib = len(tensors)
    rows_pd = [[0.0] * 6] * 6
    rows_rs = [[0.0, 0.0, 0.0]] * 6
    freqs = freqs if freqs is not None else np.linspace(520, 1700, n_vib)
    for A, f in zip(tensors, freqs):
        comp = {"xx": A[0, 0], "yy": A[1, 1], "zz": A[2, 2], "xy": A[0, 1], "xz": A[0, 2], "yz": A[1, 2]}
        rows_pd.append([comp[k] for k in order])
        a, g2 = gapmode.invariants(A)
        rows_rs.append([f, 3.7 * (45 * a * a + 7 * g2), 3 * g2 / (45 * a * a + 4 * g2)])
    n = len(rows_pd)
    txt = "$orca_hessian_file\n\n$polarizability_derivatives\n%d\n" % n
    txt += "\n".join(" ".join(f"{v: .6f}" for v in r) for r in rows_pd)
    txt += "\n\n$raman_spectrum\n%d\n" % n
    txt += "\n".join(" ".join(f"{v: .6f}" for v in r) for r in rows_rs)
    txt += "\n\n$end\n"
    path.write_text(txt)


def _sym(rng):
    M = rng.normal(size=(3, 3))
    return M + M.T


def test_gapmode_convention_and_orientation_average(tmp_path):
    rng = np.random.default_rng(3)
    T = np.stack([_sym(rng) for _ in range(12)])
    _fake_hess(tmp_path / "m.hess", T, order=("xx", "xy", "yy", "xz", "yz", "zz"))
    freqs, S, tensors, conv = gapmode.load_modes(tmp_path / "m.hess")
    assert conv["verified"] and conv["order"] == "xx-xy-yy-xz-yz-zz"
    zz = gapmode.zz_matrix(tensors, gapmode.random_rotations(20000, seed=1))
    iso = gapmode.iso_zz(tensors)
    assert np.median(np.abs(zz.mean(0) - iso) / iso) < 0.03


def test_orientation_fit_beats_isotropic_on_heldout_bands():
    rng = np.random.default_rng(5)
    T = np.stack([_sym(rng) for _ in range(16)])
    freqs = np.linspace(520, 1720, 16)
    R = gapmode.random_rotations(3000, seed=2)
    zz = gapmode.zz_matrix(T, R)
    fac = gapmode.freq_factor(freqs)
    true = 17                                        # the "measured" spectrum comes from one orientation
    grid = np.arange(500, 1751.0)
    target = gapmode.render(freqs, fac * zz[true], grid, 19.0)
    for r in gapmode.fit_orientation(freqs, zz, fac, grid, target, 19.0):
        assert r["heldout_cos_best"] > r["heldout_cos_isotropic"]
