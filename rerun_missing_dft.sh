#!/usr/bin/env bash
D=/local/scratch/mv487/dft
run_set () {   # $1 = set, $2 = comma-separated names
  for n in ${2//,/ }; do
    if [ -d "$D/$1/$n" ] && ! grep -q "ORCA TERMINATED NORMALLY" "$D/$1/$n/$n.out" 2>/dev/null; then
      mkdir -p "$D/_old"
      mv "$D/$1/$n" "$D/_old/${1}_${n}_$(date +%Y%m%d_%H%M%S)"
    fi
  done
  python dft/dft_batch.py --set "$1" --only "$2"
  python dft/dft_fix_imag.py --root "$D/$1" --set "$1" --only "$2" --fix
  python dft/dft_check.py --root "$D/$1" --set "$1"
}
run_set neutral_water asparagine
run_set neutral_gas glutamic_acid,glutamine,leucine,phenylalanine,tryptophan
