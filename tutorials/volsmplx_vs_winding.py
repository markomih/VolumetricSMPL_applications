import time
import torch
import argparse
import smplx
import numpy as np
import pickle
import tqdm
import trimesh

from VolumetricSMPL import attach_volume
from VolumetricSMPL import winding_numbers
from time import perf_counter

torch.manual_seed(0)
np.random.seed(0)

@torch.no_grad()
def visualize(model=None, smpl_output=None, scene_mesh=None, query_samples=None, collision_samples=None):
    if not VISUALIZE:
        return

    def vis_create_pc(pts, color=(0.0, 1.0, 0.0), radius=0.005):
        if torch.is_tensor(pts):
            pts = pts.cpu().numpy()

        tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
        tfs[:, :3, 3] = pts
        sm_in = trimesh.creation.uv_sphere(radius=radius)
        sm_in.visual.vertex_colors = color

        return pyrender.Mesh.from_trimesh(sm_in, poses=tfs)

    VIEWER.render_lock.acquire()
    # clear scene
    while len(VIEWER.scene.mesh_nodes) > 0:
        VIEWER.scene.mesh_nodes.pop()

    if smpl_output is not None:
        # posed_mesh = model.coap.extract_mesh(smpl_output, use_mise=True)[0]
        # posed_mesh = trimesh.Trimesh(vertices=posed_mesh.vertices, faces=posed_mesh.faces)
        posed_mesh = trimesh.Trimesh(smpl_output.vertices[0].detach().cpu().numpy(), model.faces)

        VIEWER.scene.add(pyrender.Mesh.from_trimesh(posed_mesh))

    VIEWER.scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
    if query_samples is not None:
        VIEWER.scene.add(vis_create_pc(query_samples, color=(0.0, 1.0, 0.0)))
    if collision_samples is not None:
        VIEWER.scene.add(vis_create_pc(collision_samples, color=(1.0, 0.0, 0.0)))

    VIEWER.render_lock.release()

def load_smpl_data(pkl_path):
    def to_tensor(x, device):
        if torch.is_tensor(x):
            return x.to(device=device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device)
        return x

    with open(pkl_path, 'rb') as f:
        param = pickle.load(f)
    torch_param = {key: to_tensor(val, args.device) for key, val in param.items()}
    return torch_param

@torch.no_grad()
def main():
    # Load SMPL model
    data = load_smpl_data(args.sample_body)
    scene_mesh = trimesh.load_mesh(args.scan_path)

    # create a SMPL body and attach volumetric body
    model = smplx.create(model_path=args.bm_dir_path, model_type=args.model_type, gender='neutral', use_pca=True, num_pca_comps=12, num_betas=10)
    model = attach_volume(model, pretrained=True, device=DEVICE)

    if args.model_type == 'smpl': # padd the sequence with flat hands since data is for smplx
        data['body_pose'] = torch.cat([data['body_pose'], torch.zeros_like(data['body_pose'][:, -6:])], dim=1)
    # make sure that smpl_output contains the valid SMPL variables (pose parameters, joints, and vertices). 
    assert model.joint_mapper is None, 'Volumetric Body requires valid SMPL joints as input'

    SMPL_FACES = torch.tensor(model.faces.astype(np.int64)).to(device=DEVICE)
    # benchmark the forward pass
    sync_times, max_memory = [], []
    print('Running the benchmark...')
    for step in tqdm.tqdm(range(args.num_iters)):
        # add random noise to the pose parameters
        data['body_pose'] = data['body_pose'].detach().clone() + 0.01 * torch.randn_like(data['body_pose'])
        data['betas'] = data['betas'].detach().clone() + 0.01 * torch.randn_like(data['betas'])
        smpl_output = model(**data, return_verts=True, return_full_pose=True)
        test_vertices = smpl_output.vertices.clone() + 0.1 * torch.randn_like(smpl_output.vertices) # Simulate anotehr body's vertices

        torch.cuda.reset_peak_memory_stats(device=DEVICE)
        torch.cuda.synchronize()
        start_time = perf_counter()

        # compute winding numbers
        if args.use_winding:
            winding = winding_numbers(test_vertices, smpl_output.vertices[:, SMPL_FACES])
            inside_points, outside_points = test_vertices[winding >= 0.5], test_vertices[winding < 0.5]
            # inside_points = test_vertices[winding > 0]
        else:
            sdf_values = model.volume.query(test_vertices, smpl_output)
            inside_points, outside_points = test_vertices[sdf_values <= 0], test_vertices[sdf_values > 0]

        if VISUALIZE:
            visualize(model, smpl_output, scene_mesh, outside_points, inside_points) # red: inside; green outside
            print('waiting 15 seconds')
            time.sleep(15)

        torch.cuda.synchronize()
        end_time = perf_counter()
        sync_times.append(end_time - start_time)
        max_memory.append(torch.cuda.max_memory_allocated(device=DEVICE) / 1024 / 1024 / 1024)

    sync_time = np.mean(sync_times[10:])
    max_memory = np.max(max_memory[10:])
    print(f'Average time: {sync_time * 1000} ms', f'Max memory: {max_memory} GB')
    
    if VISUALIZE:
        time.sleep(50)
        VIEWER.close_external()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Comparing the inference speed and GPU memory consumption of VolumetricSMPL vs winding numbers.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device (cuda or cpu).')
    
    # SMPL specification
    parser.add_argument('--bm_dir_path', type=str, required=False, default='../COAP_DATA/body_models', help='Directory with SMPL bodies.')
    parser.add_argument('--model_type', type=str, choices=['smpl', 'smplx'], default='smplx', help='SMPL-based body type.')
    parser.add_argument('--gender', type=str, choices=['male', 'female', 'neutral'], default='neutral', help='SMPL gender.')
    parser.add_argument('--use_winding', action='store_true', help='Use winding numbers to sample points.')
    parser.add_argument('--VISUALIZE', action='store_true', help='Use winding numbers to sample points.')

    # data samples
    parser.add_argument('--scan_path', type=str, default='./samples/scene_collision/raw_kinect_scan/scan.obj', help='Raw scan location.')
    parser.add_argument('--sample_body', type=str, default='./samples/scene_collision/sample_bodies/frame_01743.pkl', help='SMPL parameters.')

    # optimization related
    parser.add_argument('--num_iters', default=100, type=int, help='The maximum number of optimization steps.')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    args = parser.parse_args()
    DEVICE = torch.device('cuda:0')
    
    VISUALIZE = args.VISUALIZE
    if VISUALIZE:
        import pyrender
        VIEWER = pyrender.Viewer(
            pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0]), 
            use_raymond_lighting=True, run_in_thread=True, viewport_size=(1920, 1080))

    main()

# python volsmplx_vs_winding.py --bm_dir_path /media/STORAGE_4TB/COAP_DATA/body_models/ --model_type smplx
# python volsmplx_vs_winding.py --use_winding --bm_dir_path /media/STORAGE_4TB/COAP_DATA/body_models/ --model_type smplx
