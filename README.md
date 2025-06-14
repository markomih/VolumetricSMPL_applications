# VolumetricSMPL Applications

## Description
This repository provides practical examples and applications of **VolumetricSMPL**, an extension of the SMPL body model that incorporates a volumetric (signed distance field, SDF) representation. These examples demonstrate how to leverage **VolumetricSMPL** for seamless interaction with 3D geometries, including scenes, objects, and other humans.

## Installation
Ensure that PyTorch and PyTorch3D are installed with GPU support. Then, install **VolumetricSMPL** via:

```bash
pip install VolumetricSMPL
```

## Usage
VolumetricSMPL extends the interface of the [SMPL-X package](https://github.com/vchoutas/smplx) by attaching a volumetric representation to the body model. This enables functionalities such as querying signed distance fields for arbitrary points and computing collision loss terms.

### Example Usage

```python
import smplx
from VolumetricSMPL import attach_volume

# Create a SMPL body and extend it with volumetric functionalities (supports SMPL, SMPLH, and SMPL-X)
model = smplx.create(**smpl_parameters)
attach_volume(model)

# Forward pass
smpl_output = model(**smpl_data)  

# Ensure valid SMPL variables (pose parameters, joints, and vertices)
assert model.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

# Access volumetric functionalities
model.volume.query(scan_point_cloud)                 # Query SDF for given points
model.volume.selfpen_loss(smpl_output)               # Compute self-intersection loss
model.volume.collision_loss(smpl_output, scan_point_cloud)  # Compute collisions with external geometries
```

## Tutorials & Examples
The [`tutorials`](./tutorials) directory contains toy examples demonstrating how to use **VolumetricSMPL** for:
- **Human-scene interaction**: Evaluating spatial relationships between humans and environments.
- **Self-intersections**: Detecting and mitigating self-penetrations in posed bodies.

### Scene-condition Text-Driven Motion Control
To run this application presented in the paper, please follow instructions [here](https://github.com/zkf1997/DART/tree/main/VolSMPL). 

### Scene-condition Human Mesh Recovery from EgoCentric Views 
To run this application presented in the paper, please follow instructions [here](https://github.com/sanweiliti/EgoHMR). 

## Contact
For questions, please open an issue on GitHub.
