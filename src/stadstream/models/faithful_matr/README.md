# Faithful MATR Adapter

This folder is reserved for faithful MATR reproduction.

Policy:

- Keep the official MATR repository as the source of truth.
- Do not copy or rewrite MATR model, loss, decoder, or evaluation logic here.
- Only add sensor feature adapters, path/config helpers, and thin wrappers needed to call official MATR code with WiFiTAD-style sensor input.

Expected external layout:

```text
<workspace>/
  StreamSTAD/
  MATR_codebase/
```

Official repository:

```text
https://github.com/skhcjh231/MATR_codebase
```

Recommended setup on AutoDL:

```bash
cd /autodl-fs/data
git clone https://github.com/skhcjh231/MATR_codebase.git
cd /autodl-fs/data/StreamSTAD
pip install -r src/stadstream/models/faithful_matr/requirements.txt
python scripts/faithful_matr_smoke.py --matr-root ../MATR_codebase --config configs/wifitad_full.yaml
```

The smoke script only validates that official MATR can be imported and receives sensor features in its expected shape. Full faithful training/evaluation should remain in the official MATR project or call its original entrypoints.
