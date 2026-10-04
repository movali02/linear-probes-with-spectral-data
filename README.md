# probe2circuit: step 1, harmonised strings, labels, coverage, leakage

This step turns SERS spectra, QM9S spectra and ORCA DFT sticks into **one string
format produced by one code path**. It attaches structural labels from RDKit
that exist on both sides, measures how much of the SERS analyte set QM9S can
cover, and runs a suite of leakage tests before any number goes into a
write-up.

## What the model sees

```
574 0.26 19 | 605 0.61 8 | 657 0.75 18 | 962 0.42 16 | 997 0.33 16 | ...
```

Each peak is written as `position (cm-1)`, `relative intensity`, `FWHM (cm-1)`,
in ascending position order, separated by ` | `. Nothing else goes into the
string. The Laser, Substrate, Spectrum and Molecule blocks, the mode
assignments, medium, filenames and SMILES all live in record metadata and never
reach the prompt. Because the string is pure numbers, the strongest leakage
test is a **whitelist**: every input must parse under the grammar and
re-serialise byte-for-byte to itself.

## Shared processing (`configs/string_v1.yaml`)

| step | SERS | QM9S | DFT sticks |
|---|---|---|---|
| window | 500–1750 cm-1 | same | same |
| grid | 1 cm-1 | same | same |
| baseline | ALS | none (simulated) | none |
| substrate | `reference_norm`: discard peaks explained by the paired CB-only spectrum in the CB[5] regions 745–765, 812–845, 870–895 (see below). `paired` subtraction still selectable | n/a | n/a |
| smoothing | Savitzky–Golay, 9 cm-1 | same | same |
| normalisation | CB[5] ~829 band = 1.0 (analyte peaks can exceed 1.0) | max **inside the window** (the old code normalised over 400–4000, so C–H stretches set the max) | same as QM9S |
| peaks | prominence ≥ max(0.03, 4σ_noise), height ≥ 0.05, FWHM ≥ 5 cm-1, top 40 | same | same |
| width | FWHM (rel_height 0.5) | same (old QM9S code used 0.7) | same |

Changing any value means bumping `schema`, because strings made under
different settings are no longer comparable.

### SERS substrate: `reference_norm` (schema v1.1)

1. **Normalise.** Each SERS spectrum is divided by its CB[5] reference band: the
   tallest real peak within 829 ± 4 cm-1 (the band sits at 826–827 in the cell
   session and 828–830 in the standards). That band is 1.0, and every other
   intensity is its ratio to it. Nothing is subtracted, so there are no negative
   intensities, and a spectrum recorded 5× brighter gives the same string.
2. **Discard substrate peaks.** The paired CB-only spectrum (mean of the CB
   repeats for the same strain+medium) is put on the same 829 = 1.0 scale. Its
   bands inside `cb_regions` give the expected substrate height at each
   position. A peak within 5 cm-1 of such a band is dropped **unless** it is
   clearly taller: ≥ 0.10 above it **and** ≥ 1.5× it. That rule, not the
   position, is what keeps Trp/indole 758 on top of CB ~755 and Trp 877 on top
   of CB ~880. The two are 3–4 cm-1 apart with ~19 cm-1 lines, so the detector
   sees one merged peak. A kept peak is reported at its full observed height.
3. **Reference band.** The ~829 band itself is always dropped (it is 1.0 by
   construction).

Per-record audit trail in `substrate`: `ref_height`, `ref_pos_cm`, `cb_ref`,
`cb_bands`, `discarded` ([position, intensity, CB expected]).

Fallbacks are flagged, never silent:
- `cb_ref: "global_mean"`: no CB-only file for that condition, so the mean over all
  sessions is used. The band positions can be 2–3 cm-1 off.
- `mode_used: "max_in_window(ref_missing)"`: no real peak at ~829 (prominence
  < 2% of the spectrum max), so the spectrum is max-normalised and nothing is
  discarded.

Caveats:
- **Tyrosine.** Tyr 830 sits on the reference band. It inflates the divisor, so a
  tyrosine spectrum's other peaks come out smaller. Its ~830 signal is always
  discarded with the reference; Tyr 850 is kept.
- **617 and 676 CB bands are not in `cb_regions`,** so they reach every string,
  including CB-only ones. Add `[600, 635], [665, 690]` to `cb_regions` to drop
  them; the same taller-than-substrate rule protects analyte peaks there.
- **Intensity scale differs from QM9S/DFT** (829-relative vs strongest-peak).
  For QM9S→SERS transfer, set `rescale_to_max_after_discard: true`: 829 still
  sets the detection thresholds, then the strongest analyte peak becomes 1.0.
- `min_height` / `min_prominence` are now in 829 units for SERS. If the
  analyte dwarfs the CB band, weak analyte peaks sit well above threshold. If
  the CB band dwarfs the analyte, they can fall below it.

## Labels (`labels.py`)

The labels are 23 SMARTS functional groups plus `n_heavy`, `elements`, ring
counts, `has_S`, `has_isotope` and `in_qm9_domain`. Matching ignores stereo and
isotopes, so indole-d6 matches indole as a structure but is still out of domain
spectrally. Analyte SMILES come from your Excel. The five non-amino-acids at the
bottom of `data/analytes.csv` (`added_verify`) were added by me, so check them.

## Running it

```bash
conda env create -f environment.yml && conda activate probe2circuit
pytest                                    # 52 tests, CPU, ~2 s

# preferred: QM9S from the broadened spectra (window-first normalisation).
# Symlink the real files over the placeholders in data/qm9s/ first; see RUN_ON_AURA.md
bash scripts/run_step1.sh
# fallback: QM9S from make_views.py peak lists (what the 25 Sep run used)
SERS_DIR=... QM9S_JSONL=.../qm9s_raman.jsonl bash scripts/run_step1.sh
# exact token counts on Aura: add TOKENIZER=Qwen/Qwen2.5-7B-Instruct
```

Outputs:
- `data/processed/{sers,qm9s}_records.jsonl`
- `reports/coverage.md` (+ CSVs)
- `reports/leakage_sers.md`

`run_leakage_checks.py` exits 1 if any check fails, so it can gate the
fine-tuning script.

## Leakage checks (`leakage.py`)

| check | fails when |
|---|---|
| `grammar_whitelist` | any input is not a pure peak list that round-trips |
| `forbidden_terms` | an analyte name, 3-letter code, SMILES, formula, file id, or "M9"/"medium" appears in an input |
| `prompt_template` | the fixed instruction contains a label term |
| `split_disjoint` | an id is duplicated, or a group sits in both train and test |
| `duplicates` | the same sample (`--identity_key`: analyte for SERS, SMILES for QM9S) appears with an identical string in both splits. Identical strings for *different* samples, and near-duplicates at cosine ≥ 0.98, are warnings |
| `token_budget` | prompt + answer reserve > `max_seq_length` (exact with `--tokenizer`, estimated otherwise) |
| `shuffled_label_control` | LR on shuffled labels beats chance; also reports the **real LR baseline** and a permutation p-value |
| `substrate_only_control` | a classifier trained on analyte spectra predicts the "label" of CB-only spectra above chance, meaning it is reading session, day or substrate, not the analyte |

Each check has a test that injects the leak and confirms it fails.

## What the 25 Sep run shows (`reports/run_20260925/`)

Inputs: your 7 SERS files, and `qm9s_raman.jsonl` (125,996 molecules) used both
as the QM9 reference and as the QM9S spectra.

### QM9S → SERS: simpler explanations than symmetry

1. **Most QM9S fingerprint information is missing from the file you trained on.**
   - The peak lists were normalised to the strongest peak over 400–4000 cm-1 (usually a C–H, N–H or O–H stretch) and then cut at 0.1.
   - Only 59,010 of the 125,996 molecules (47%) keep any peak in 500–1750.
   - Among those, the median is **1 peak**; 29,733 have exactly one.
   - Inside the window, a peak survives only if it's above 0.62 of the strongest fingerprint peak (median).
   - Only 15,473 molecules have ≥ 3 peaks in the window. SERS strings have 8–15.
   - A model trained on these strings mostly learned the high-wavenumber stretches, and SERS has none of them in its window.
2. **Many window strings can't tell molecules apart.** 27,665 of the 59,010 window strings (47%) are identical to another molecule's string. The most common is `1664 1.00 10`, shared by 130 molecules.
3. **The chemistry doesn't overlap.**
   - **This QM9S file has no carboxylic acids at all**, and no zwitterions. Every amino acid is a zwitterion in water, and the COO- bands (~1410 cm-1) have nothing to match on the QM9S side. It's worth checking whether any acids were dropped when the file was built: it has 126k molecules, and QM9 has about 134k.
   - None of your 25 analytes is in QM9S except indole and indole-d6.
   - Thiol and thioether have zero support. The indole group appears in exactly 1 molecule (indole itself), and the benzene ring in 89 usable molecules.
4. **The QM9S broadening is 10 cm-1 FWHM (inferred).** Isolated peaks in the file measure 15.3 cm-1 at rel_height 0.7, which is a 10.0 cm-1 Lorentzian. SERS FWHM is about 19 cm-1 median.
5. **Aromaticity is barely decodable from the QM9S window strings.** A logistic regression reaches 0.80 against a 0.72 majority baseline.

Before invoking SO(3) vs C∞, rebuild QM9S from `raman_boraden.csv` with the
window-first normalisation (`--qm9s_csv`; the code is ready). Then compare only
the labels with real support.

### SERS
6. **Old strings vs new strings, per spectrum** (table in `leakage_sers.md`):
   - The old strings carry the label (medium, Molecule block, name-derived mode assignments).
   - They included 3–5 CB[5] substrate peaks each.
   - They were about 1,000–1,300 tokens; the new ones are 110–220.
7. **The standards are dominated by a shared background.**
   - Bands near 671 and 1601 cm-1 appear in all 5 standards; 1286 and 1377 in 4.
   - 671 and 1286 are also in the CB-only spectrum.
   - The CB band sits at 826–827 cm-1 in the cell session and at 828–830 in the standards.
   - Subtracting a reference across sessions is unsafe, so `paired` never does it.
8. **Masking destroys real bands** (Trp 759 overlaps CB 755, Tyr 830 overlaps CB 827). It's a fallback only.
   `reference_norm` (Sep 26) replaces fixed-tolerance masking with the taller-than-substrate rule above for this reason.

### Decisions
- **Substrate pairing** defaults to the mean of all CB repeats for the same strain and medium (`CB_536_trp_*` for `Cell_536_trp_*`). `pair_by: same_index` uses the matching repeat instead.
- **Your detect_peaks v2 is not the shared detector.** Its second-derivative pass adds shoulder peaks. QM9S was peak-picked without that pass, so SERS strings would get systematically more peaks than QM9S strings of the same chemistry. The noise-based prominence threshold is kept.
- The old `Cell_CB_*` test rows are substrate-only spectra. They belong in `substrate_only_control`, never in train or test.

## Known limitations / open items

- QM9S strings from `qm9s_raman.jsonl` are a stopgap (see `effective_floor` per record). Rebuild them from `raman_boraden.csv`.
- Set `sim_extra_fwhm` to about 13 cm-1 (Gaussian), which turns the 10 cm-1 QM9S Lorentzian into a ~19 cm-1 Voigt like SERS, but only once QM9S is rebuilt from spectra. It can't be applied to peak lists.
- Splits are grouped by `id` for now, as agreed. `duplicates` warns if replicate scans end up on both sides.
- The CB-only spectra for the standards aren't in the repo yet. Their filenames need to match `pair_by`.
