## Train normal generation

```bash
python causaldiff_repro/run.py train-normal \
  --data path/to/data.csv \
  --output causaldiff_repro/outputs/energy \
  --window-size 48 \
  --max-lag 8 \
  --predictor tcn \
  --evaluate --quick-eval
```

Outputs include:

- `predictor.pt`
- `kan_mechanism.pt`
- `normal_diffusion.pt`
- `causal_tensors.pt`
- `representations.pt`
- `generated_normal.npz`
- `metrics.json`

## Train anomalous generation

```bash
python causaldiff_repro/run.py train-shift \
  --normal-dir causaldiff_repro/outputs/energy \
  --normal-data path/to/normal.csv \
  --anomaly-data path/to/anomaly.csv \
  --output causaldiff_repro/outputs/energy_anomaly \
  --samples 256
```

If a label column is provided, label `0` is treated as normal and labels `>0` as anomalous.
