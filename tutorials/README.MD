# VolSMPL vs winding numbers

### install VolSMPL:
Script `volsmplx_vs_winding.py` contains comparison (runtime and GPU memory)
```bash
# To run inference with VolSMPL:
python volsmplx_vs_winding.py --bm_dir_path ${MODELS} --model_type smplx

# To run inference with winding numbers add flag --use_winding:
python volsmplx_vs_winding.py --bm_dir_path ${MODELS} --model_type smplx --use_winding

# To visually debug that everything is running correctly, you could add --VISUALIZE flag
```

# Tutorials

We provide example scripts to use VolumeSMPL inside of optimization loops to resolve collisions with environments and self-intersections. 

### **Collisions with Environment**:
Execute the following script to load a scene (scene mesh and a SMPL body) and optimize the placement of the human body such that collisions are resolved:
```bash
python scene_collisions.py --bm_dir_path  ${MODELS} --model_type smplx

# To visually debug that everything is running correctly, you could add --VISUALIZE flag
```
where `${MODELS}` is a directory path with the SMPL bodies of the following structure (for more details see https://github.com/vchoutas/smplx#model-loading):
```bash
${MODELS}
├── smpl
│   ├── SMPL_FEMALE.pkl
│   └── SMPL_MALE.pkl
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
![scene-collisions](./samples/static/scenecol.png)
