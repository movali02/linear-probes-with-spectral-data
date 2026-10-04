# Running step 1 on Aura

It runs on CPU only and takes roughly 10–15 minutes, most of it reading the
QM9S CSV. You don't need a GPU node.

## 1. Copy and unpack

```bash
# from your laptop
scp probe2circuit_step1.zip <you>@<aura-login-node>:~/
# on Aura
unzip probe2circuit_step1.zip && cd probe2circuit
```

## 2. Environment

Use your existing env and add what's missing:

```bash
conda activate mv_env
pip install -e .                 # numpy, scipy, pandas, scikit-learn, pyyaml, rdkit, openpyxl
pip install torch_geometric      # only needed once, to read qm9s.pt
pytest -q                        # 39 tests, ~2 s: confirms the install
```

## 3. Point the placeholders at the real files

`data/qm9s/raman_boraden.csv` and `data/qm9s/qm9s.pt` are placeholders. Replace
them with symlinks, so nothing large is copied:

```bash
ln -sf /path/to/qm9s/raman_boraden.csv data/qm9s/raman_boraden.csv   # note the dataset's own spelling
ln -sf /path/to/qm9s/qm9s.pt           data/qm9s/qm9s.pt
```

Put every SERS spectrum in `data/raw_sers/`, or replace the folder with a
symlink to where the spectra already live. Use the raw instrument exports, like
the `Cell_BW_tyr_1.txt` you sent, not normalised copies.

| file | meaning |
|---|---|
| `Cell_<strain>_<aa>_<n>.txt` | cell spectra |
| `CB_<strain>_<aa>_<n>.txt` | substrate-only spectra for those cells |
| `L-<name>.txt` | standards |
| `CB_L-<name>_<n>.txt` | substrate-only spectra for the standards |

Old-style names such as `Cell_Cell_…` and `Cell_CB_…` are handled.

## 4. Check, then run

```bash
python scripts/preflight.py --tokenizer /path/to/Qwen2.5-7B-Instruct   # local model dir if the node has no internet
TOKENIZER=/path/to/Qwen2.5-7B-Instruct bash scripts/run_step1.sh
```

`run_step1.sh` runs the preflight itself and stops before any slow step if
something is missing. Read the preflight output before the full run:
- **FAIL**: must be fixed, e.g. a placeholder not yet replaced, or missing torch_geometric.
- **WARN**: runs anyway, but the result is weaker, e.g. spectra without a CB reference fall back to masking.

To submit it as a job instead: `sbatch --wrap "bash scripts/run_step1.sh" -c 4 --mem 16G -t 1:00:00`,
plus whatever partition flag Aura needs.

## 5. Send back

Zip `reports/run_<date>/` (a few hundred KB) and attach it here, along with the
terminal output. `data/processed/` is regenerated each run and is large, so
don't send it.
