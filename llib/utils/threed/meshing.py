import smplx
from quad_mesh_simplify import simplify_mesh
import os
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
import numpy as np 
import trimesh
import torch
import os.path as osp
import pickle
import os

"""
Script to create low resolution meshes of SMPL-X
"""

essentials_folder = 'essentials/'
body_model_utils_folder = essentials_folder + 'body_model_utils'
model_folder = essentials_folder + 'body_models'
model_type = 'smplx'

bm = smplx.create(
    model_path=model_folder,
    model_type=model_type
)
faces = bm.faces 
faces = torch.load(
    osp.join(body_model_utils_folder, f'{model_type}_faces.pt')
)
vertices = bm().vertices.detach()

if model_type == 'smplx':
    # remove eye balls 
    eye_balls = trimesh.load(
        essentials_folder + 'selfcontact-essentials/segments/smplx/smplx_segment_headwoeyes.ply',
        process=False)
    eyeball_col = np.array([150, 150, 150, 255])
    eyeball_vids = np.where(np.all(eye_balls.visual.vertex_colors == eyeball_col, axis=1))[0]

    # in smpl-x the eyeball vertices are the last 1092 vertices
    vertices = vertices[:,:eyeball_vids.min(),:]

    # remove all faces that contain eyeball vertices
    faces_noeyeball = faces[~np.any(np.isin(faces.cpu().numpy(), eyeball_vids), axis=1)]
    mesh = trimesh.Trimesh(vertices[0].cpu().numpy(), faces_noeyeball.cpu().numpy(), process=False)
    print(mesh.is_watertight)
    _ = mesh.export('../../../outdebug/smplx_noeyeball.obj')

    # close back of mouth
    # add vertex
    inner_mouth_verts_path = pickle.load(
        open(f'{body_model_utils_folder}/smplx_inner_mouth_bounds.pkl', 'rb')
    )
    # reverse list inner_mouth_verts_path so that normals point outwards
    inner_mouth_verts_path = inner_mouth_verts_path[::-1]
    vert_ids_wt = torch.tensor(inner_mouth_verts_path)
    mouth_vert = torch.mean(vertices[:,vert_ids_wt,:], 1, keepdim=True)
    vertices = torch.cat((vertices, mouth_vert), 1).detach().cpu().numpy()
    # add faces
    max_face_id = faces_noeyeball.max().item() + 1
    faces_mouth_closed = [] # faces that close the back of the mouth
    for i in range(len(vert_ids_wt)-1):
        faces_mouth_closed.append([vert_ids_wt[i], vert_ids_wt[i+1], max_face_id])
    faces_mouth_closed = torch.tensor(np.array(faces_mouth_closed).astype(np.int64), dtype=torch.long, device=faces.device)
    faces = torch.cat((faces_noeyeball, faces_mouth_closed), 0)

    mesh = trimesh.Trimesh(vertices[0], faces.detach().cpu().numpy())
    _ = mesh.export('../../../outdebug/smplx_noeyeball_mouthclosed.obj')
    mesh.is_watertight # True


low_res = {}
for nn in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:

    new_verts, new_faces = simplify_mesh(vertices[0].astype(np.float64), faces.cpu().numpy().astype(np.uint32), nn)

    # repair mesh
    mesh = trimesh.Trimesh(new_verts, new_faces)
    broken_faces_idx = trimesh.repair.broken_faces(mesh, color=[255,0,0,255])
    #_ = mesh.export('../../../outdebug/broken.obj')
    broken_faces_idx = trimesh.repair.broken_faces(mesh)
    if len(broken_faces_idx) > 0:
        broken_faces_sorted = np.sort(new_faces[broken_faces_idx], axis=1)
        vals, idx, count = np.unique(broken_faces_sorted, axis=0, return_index=True, return_counts=True)
        actually_broken_faces_sorted = vals[count > 1]
        bool_mask = [True if x in actually_broken_faces_sorted else False for x in broken_faces_sorted]
        actually_broken_faces_mask = np.array([np.any(np.all(actually_broken_faces_sorted == x, axis=1)) for x in broken_faces_sorted])
        actually_broken_faces = broken_faces_idx[actually_broken_faces_mask]
        faces_idxs = np.array([i for i in range(new_faces.shape[0]) if i not in actually_broken_faces])
        faces_repaired = new_faces[faces_idxs]
    else:
        faces_repaired = new_faces

    # visualize
    mesh = trimesh.Trimesh(new_verts, faces_repaired, process=False)
    print(nn, mesh.is_watertight)
    _ = mesh.export(f'{body_model_utils_folder}/lowres_smplx_{nn}vertices.obj')

    # the new vertices must not be on the mesh surface, so we find the closest point on the smpl-x mesh surface
    distance = pcl_pcl_pairwise_distance(
        x=torch.from_numpy(new_verts).unsqueeze(0).double(),
        y=torch.from_numpy(vertices).double(),
    )
    closest_vert_ids = torch.argmin(distance, dim=2).squeeze(0).cpu().numpy()

    distance = pcl_pcl_pairwise_distance(
        x=torch.from_numpy(new_verts).unsqueeze(0).double(),
        y=torch.from_numpy(vertices).double(),
    )
    closest_vert_ids = torch.argmin(distance, dim=2).squeeze(0).cpu().numpy()
    # check if the closest vertices is the appended back of the mouth vertex. In this case we need to reindex the vertex id.
    if (closest_vert_ids == 9383).sum() > 0:
        bmidx = np.where(closest_vert_ids == 9383)[0]
        closest_vert_ids[bmidx] = bm().vertices.shape[1]

    low_res[nn] = {'faces': faces_repaired, 'smplx_vid': closest_vert_ids, 'is_watertight': mesh.is_watertight}

# visualize poses with new faces
example_folder = osp.join(essentials_folder, 'selfcontact-essentials/example_meshes/selfcontact/')
test_meshes = os.listdir(example_folder)
for tm in test_meshes:
    if tm.startswith('.'):
        continue
    mesh = trimesh.load(osp.join(example_folder, tm), process=False)
    vertices = torch.from_numpy(np.array(mesh.vertices)).unsqueeze(0)
    mouth_vert = torch.mean(vertices[:,vert_ids_wt,:], 1,
            keepdim=True)
    vertices = torch.cat((vertices, mouth_vert), 1).detach().cpu().numpy()
    for nn, ii in low_res.items():
        new_verts = vertices[0, ii['smplx_vid'], :]
        new_mesh = trimesh.Trimesh(new_verts, ii['faces'], process=False)
        print(tm, nn, new_mesh.is_watertight)
        outfn = f'{body_model_utils_folder}/lowres/lowres_smplx_{tm}_{nn}_vertices.ply'
        _ = new_mesh.export(outfn)

# save low_res data as pickle file
with open(f'{body_model_utils_folder}/lowres_smplx.pkl', 'wb') as f:
    pickle.dump(low_res, f)