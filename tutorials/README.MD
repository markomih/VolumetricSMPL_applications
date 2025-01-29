# Tutorials & Examples for VolumetricSMPL

## Comparison: VolumetricSMPL vs Winding Numbers
The script `volsmplx_vs_winding.py` benchmarks **VolumetricSMPL** against **winding numbers**, evaluating **runtime** and **GPU memory usage**.

```bash
# Run inference with VolumetricSMPL:
python volsmplx_vs_winding.py --bm_dir_path ${MODELS} --model_type smplx

# Run inference using winding numbers:
python volsmplx_vs_winding.py --bm_dir_path ${MODELS} --model_type smplx --use_winding

# Add the --VISUALIZE flag for debugging:
python volsmplx_vs_winding.py --bm_dir_path ${MODELS} --model_type smplx --VISUALIZE
```

---

## Resolving Collisions with the Environment
The script `scene_collisions.py` optimizes the placement of a **SMPL body** inside a **scene mesh** to **minimize collisions**.

### Run the optimization:
```bash
python scene_collisions.py --bm_dir_path ${MODELS} --model_type smplx

# Add the --VISUALIZE flag to debug the optimization process:
python scene_collisions.py --bm_dir_path ${MODELS} --model_type smplx --VISUALIZE
```

---

## Model Directory Structure
Set `${MODELS}` to the directory containing **SMPL**, **SMPLH**, and **SMPL-X** models.  
The expected directory structure is:

```bash
${MODELS}
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smplh
│   ├── SMPLH_FEMALE.pkl
│   └── SMPLH_MALE.pkl
└── smplx
    ├── SMPLX_FEMALE.npz
    ├── SMPLX_FEMALE.pkl
    ├── SMPLX_MALE.npz
    ├── SMPLX_MALE.pkl
    ├── SMPLX_NEUTRAL.npz
    └── SMPLX_NEUTRAL.pkl
```

For more details, refer to the [SMPL-X model loading guide](https://github.com/vchoutas/smplx#model-loading).

---

## Example Visualization *(under construction)*