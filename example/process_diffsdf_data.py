import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import csv
import json
import h5py
import trimesh
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels, mesh_to_sdf
from mesh_to_sdf.utils import scale_to_unit_sphere


sdf_data_path = 'diffsdf_data/acronym/Couch/37cfcafe606611d81246538126da07a8/sdf_data.csv'
grid_gt_path = 'diffsdf_data/grid_data/acronym/Couch/37cfcafe606611d81246538126da07a8/grid_gt.csv'
splits_path = 'diffsdf_data/splits/couch_all.json'
target_obj_path = 'example/chair.obj'
target_dataset = 'Acronym' # 'Acronym', 'debug'
acronym_dataset_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/acronym/grasps'
# save_path = ''


def save_obj_with_color(v, f=None, c=None, file_name=''):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        if c is None:
            obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
        else:
            obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + ' ' + str(c[i][0]) + ' ' + str(c[i][1]) + ' ' + str(c[i][2]) + '\n')
    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def load_acronym_dataset(shapenetsem_path):
    # returns list of trimesh files for Acronym objects
    acronym_mesh_list = []
    for each_h5_file in os.listdir(acronym_dataset_path):
        grasps = h5py.File(os.path.join(acronym_dataset_path, each_h5_file), 'r')
        _, obj_name, obj_file_path = grasps['object/file'][()].decode('utf-8').split('/')
        obj_file_path = os.path.join(shapenetsem_path, obj_file_path)
        each_h5_mesh = trimesh.load(obj_file_path, force='mesh')
        # save_obj_with_color(v=each_h5_mesh.vertices, f=each_h5_mesh.faces, c=None, file_name='debug_mesh_h5.obj')
        acronym_mesh_list.append(each_h5_mesh)
    return acronym_mesh_list


# Choose target dataset
if target_dataset == 'Acronym':
    # Load Acronym dataset
    print("Loading Acronym dataset....")
    shapenetsem_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/ShapeNetSem/data/models-OBJ/models'
    acronym_mesh_list = load_acronym_dataset(shapenetsem_path)
    target_mesh_list = acronym_mesh_list
elif target_dataset == 'debug':
    mesh = trimesh.load(target_obj_path)
    target_mesh_list = [mesh]
else:
    raise AttributeError('Not implemented!!!')


# Load Diffusion SDF sample files for reference
print("Loading sample files from Diffusion SDF....")
with open(sdf_data_path, newline='') as f: # sdf_data: Points on and near the mesh
    reader = csv.reader(f)
    sdf_data = np.array(list(reader)).astype(float) # [596000, 4]
with open(grid_gt_path, newline='') as f: # grid_gt: SDF values for points distributed on a grid, and are intended to sample empty areas
    reader = csv.reader(f)
    grid_gt = np.array(list(reader)).astype(float) # [468000, 4]
    grod_gt_coords = grid_gt[:, :3]
with open(splits_path) as f:
    splits = json.load(f)


for each_target_mesh in target_mesh_list:
    # Extract sdf_data
    print("Extracting SDFs from our own datasets....")
    mesh = each_target_mesh
    ext_points, ext_sdf = sample_sdf_near_surface(mesh, number_of_points=len(sdf_data)) # points: [596000, 3], sdf: [596000]
    ext_sdf_data = np.concatenate([ext_points, ext_sdf[:, None]], axis=-1)
    ext_voxels = mesh_to_voxels(mesh, 128, pad=False, sign_method='normal') ####### Changed from pad=True to pad=False

    # Extract grid_gt
    grid_gt_query_points = grod_gt_coords
    mesh = scale_to_unit_sphere(mesh)
    ext_grid_gt_sdf = mesh_to_sdf(mesh, grid_gt_query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
    ext_grid_gt = np.concatenate([grid_gt_query_points, ext_grid_gt_sdf[:, None]], axis=-1)
    filtered_grid_gt_query_points = grid_gt_query_points[ext_grid_gt_sdf < 0] # optional for visualization purpose


    # Visualization
    print("Visualizing SDFs from our own datasets....")
    sdf_data_colors = np.zeros(ext_points.shape)
    sdf_data_colors[ext_sdf < 0, 2] = 1
    sdf_data_colors[ext_sdf > 0, 0] = 1

    grid_gt_colors = np.zeros(grid_gt_query_points.shape)
    grid_gt_colors[ext_grid_gt_sdf < 0, 2] = 1
    grid_gt_colors[ext_grid_gt_sdf > 0, 0] = 1

    save_obj_with_color(v=mesh.vertices, f=mesh.faces, c=None, file_name='debug_mesh.obj')
    save_obj_with_color(v=ext_points, f=None, c=sdf_data_colors, file_name='debug_pc.obj')
    save_obj_with_color(v=grod_gt_coords, f=None, c=None, file_name='debug_grid_gt.obj')
    save_obj_with_color(v=grid_gt_query_points, f=None, c=grid_gt_colors, file_name='debug_ext_grid_gt.obj')
    save_obj_with_color(v=filtered_grid_gt_query_points, f=None, c=None, file_name='debug_ext_grid_gt_filter.obj')


    # Save
    import pdb; pdb.set_trace()