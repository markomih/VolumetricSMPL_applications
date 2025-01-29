import time
import pickle
import argparse
import smplx
import torch
import trimesh
import pyrender
import numpy as np
import torch.nn.functional as F

from VolumetricSMPL import attach_volume

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
        # posed_mesh = model.volume.extract_mesh(smpl_output, use_mise=True)[0]
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
def sample_scene_points(model, smpl_output, scene_vertices, scene_normals=None, n_upsample=2, max_queries=10000):
    points = scene_vertices.clone()
    # remove points that are well outside the SMPL bounding box
    bb_min = smpl_output.vertices.min(1).values.reshape(1, 3)
    bb_max = smpl_output.vertices.max(1).values.reshape(1, 3)

    inds = (scene_vertices >= bb_min).all(-1) & (scene_vertices <= bb_max).all(-1)
    if not inds.any():
        return None
    points = scene_vertices[inds]
    return points.float().reshape(1, -1, 3)  # add batch dimension


def main():
    # load data sample
    data = load_smpl_data(args.sample_body)
    scene_mesh = trimesh.load_mesh(args.scan_path)

    # create a SMPL body and attach volumetric body
    model = smplx.create(model_path=args.bm_dir_path, model_type=args.model_type, gender='neutral', use_pca=True, num_pca_comps=12, num_betas=10)
    model = attach_volume(model, pretrained=True, device=args.device)
    
    scene_vertices = torch.from_numpy(scene_mesh.vertices).to(device=args.device, dtype=torch.float)
    scene_normals = torch.from_numpy(np.asarray(scene_mesh.vertex_normals).copy()).to(device=args.device, dtype=torch.float)

    # visualize
    if args.model_type == 'smpl': # padd the sequence with flat hands since data is for smplx
        data['body_pose'] = torch.cat([data['body_pose'], torch.zeros_like(data['body_pose'][:, -6:])], dim=1)
    smpl_output = model(**data, return_verts=True, return_full_pose=True)
    # NOTE: make sure that smpl_output contains the valid SMPL variables (pose parameters, joints, and vertices). 
    assert model.joint_mapper is None, 'requires valid SMPL joints as input'

    visualize(model, smpl_output, scene_mesh)
    print('waiting 5 seconds')
    time.sleep(5)
    
    # create an optimizer
    init_pose = data['body_pose'].detach().clone()
    params_to_optimize = ['transl']#, 'global_orient'
    for param in params_to_optimize:
        data[param].requires_grad = True
    opt = torch.optim.Adam([data[param] for param in params_to_optimize], lr=args.lr)
    for step in range(args.max_iters):
        # smpl forward pass
        smpl_output = model(**data, return_verts=True, return_full_pose=True)
        
        # compute self collision loss
        scene_points = sample_scene_points(model, smpl_output, scene_vertices, scene_normals)
        if scene_points is None:
            print('No more colliding points')
            break
        selfpen_loss, _collision_mask = model.volume.collision_loss(scene_points, smpl_output, ret_collision_mask=True)

        # visualization and opt step 
        if VISUALIZE:
            _collision_samples = scene_points[_collision_mask]
            _non_collision_samples = scene_points[~_collision_mask]
            visualize(model, smpl_output, scene_mesh, _non_collision_samples, _collision_samples)
            print('iter ', step, ':\t', selfpen_loss, '\tWaiting 0.5s')
            time.sleep(0.5)
        
        if selfpen_loss < 0.5:
            print('Converged')
            break
        opt.zero_grad()
        selfpen_loss.backward(retain_graph=True)
        print(data['transl'], data['transl'].grad)
        opt.step()

    visualize(model, smpl_output, scene_mesh)
    print('exiting in 10 seconds')
    time.sleep(10)
    if VISUALIZE:
        VIEWER.close_external()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tutorial on how to use VolumetricSMPL to avoid collisions with static geometries.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device (cuda or cpu).')
    
    # SMPL specification
    parser.add_argument('--bm_dir_path', type=str, required=False, default='../COAP_DATA/body_models', help='Directory with SMPL bodies.')
    parser.add_argument('--model_type', type=str, choices=['smpl', 'smplx'], default='smplx', help='SMPL-based body type.')
    parser.add_argument('--gender', type=str, choices=['male', 'female', 'neutral'], default='neutral', help='SMPL gender.')
    parser.add_argument('--VISUALIZE', action='store_true', help='Use winding numbers to sample points.')

    # data samples
    parser.add_argument('--scan_path', type=str, default='./samples/scene_collision/raw_kinect_scan/scan.obj', help='Raw scan location.')
    parser.add_argument('--sample_body', type=str, default='./samples/scene_collision/sample_bodies/frame_01743.pkl', help='SMPL parameters.')

    # optimization related
    parser.add_argument('--max_iters', default=200, type=int, help='The maximum number of optimization steps.')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    args = parser.parse_args()
    
    VISUALIZE = args.VISUALIZE
    if VISUALIZE:
        VIEWER = pyrender.Viewer(
            pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0]), 
            use_raymond_lighting=True, run_in_thread=True, viewport_size=(1920, 1080))

    main()

# python scene_collisions.py --bm_dir_path  /media/STORAGE_4TB/COAP_DATA/body_models/ --model_type smplx 
# python scene_collisions.py --bm_dir_path  /media/STORAGE_4TB/COAP_DATA/body_models/ --model_type smplx --VISUALIZE
