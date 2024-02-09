import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import csv
import json
import h5py
import time
import trimesh
import subprocess
import numpy as np
import pandas as pd 
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels, mesh_to_sdf
from mesh_to_sdf.utils import scale_to_unit_sphere


sdf_data_path = 'diffsdf_data/acronym/Couch/37cfcafe606611d81246538126da07a8/sdf_data.csv'
grid_gt_path = 'diffsdf_data/grid_data/acronym/Couch/37cfcafe606611d81246538126da07a8/grid_gt.csv'
splits_path = 'diffsdf_data/splits/couch_all.json'
target_obj_path = 'example/chair.obj'
target_dataset = 'Acronym' # 'ShapeNetSem', 'Acronym', 'ObMan', 'DexYCB', 'debug'
acronym_dataset_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/acronym/grasps'
create_watertight = False
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


def create_watertight_acronym(shapenetsem_path, acronym_watertight_path, simple_acronym_watertight_path):
    # Watertight
    for each_h5_file in os.listdir(acronym_dataset_path):
        grasps = h5py.File(os.path.join(acronym_dataset_path, each_h5_file), 'r')
        _, obj_name, obj_file_path = grasps['object/file'][()].decode('utf-8').split('/')
        obj_file_name = obj_file_path.split('.obj')[0]
        print(f'Processing {obj_file_name}....')
        full_obj_file_path = os.path.join(shapenetsem_path, obj_file_path)
        out_file_path = os.path.join(acronym_watertight_path, obj_file_path)
        subprocess.call(["externals/Manifold/build/manifold", full_obj_file_path, out_file_path, "-s"])
    # # Simplify
    # for each_h5_file in os.listdir(acronym_dataset_path):
    #     grasps = h5py.File(os.path.join(acronym_dataset_path, each_h5_file), 'r')
    #     _, obj_name, obj_file_path = grasps['object/file'][()].decode('utf-8').split('/')
    #     obj_file_name = obj_file_path.split('.obj')[0]
    #     print(f'Processing {obj_file_name}....')
    #     full_obj_file_path = os.path.join(shapenetsem_path, obj_file_path)
    #     out_file_path = os.path.join(acronym_watertight_path, obj_file_path)
    #     out_simple_file_path = os.path.join(simple_acronym_watertight_path, obj_file_path)
    #     subprocess.call(["externals/Manifold/build/simplify", "-i", out_file_path, "-o", out_simple_file_path, "-m", "-r", "0.02"])


def create_watertight_shapenet(shapenetsem_path, shapenet_watertight_path, simple_shapenet_watertight_path):
    # Watertight
    for obj_file_path in [f for f in os.listdir(shapenetsem_path) if '.obj' in f]:
        obj_file_name = obj_file_path.split('.obj')[0]
        print(f'Processing {obj_file_name}....')
        full_obj_file_path = os.path.join(shapenetsem_path, obj_file_path)
        out_file_path = os.path.join(shapenet_watertight_path, obj_file_path)
        subprocess.call(["externals/Manifold/build/manifold", full_obj_file_path, out_file_path, "-s"])
    # # Simplify
    # for obj_file_path in [f for f in os.listdir(shapenetsem_path) if '.obj' in f]:
    #     obj_file_name = obj_file_path.split('.obj')[0]
    #     print(f'Processing {obj_file_name}....')
    #     full_obj_file_path = os.path.join(shapenetsem_path, obj_file_path)
    #     out_file_path = os.path.join(shapenet_watertight_path, obj_file_path)
    #     out_simple_file_path = os.path.join(simple_shapenet_watertight_path, obj_file_path)
    #     subprocess.call(["externals/Manifold/build/simplify", "-i", out_file_path, "-o", out_simple_file_path, "-m", "-r", "0.02"])


def load_acronym_dataset(shapenetsem_path):
    # returns list of trimesh files for Acronym objects
    acronym_mesh_list = []
    # for each_h5_file in os.listdir(acronym_dataset_path):
    for each_h5_file in os.listdir(acronym_dataset_path)[:4000]:
        grasps = h5py.File(os.path.join(acronym_dataset_path, each_h5_file), 'r')
        _, obj_name, obj_file_path = grasps['object/file'][()].decode('utf-8').split('/')
        obj_file_name = obj_file_path.split('.obj')[0]
        print(f'Processing {obj_file_name}....')
        full_obj_file_path = os.path.join(shapenetsem_path, obj_file_path)
        watertight_obj_file_path = os.path.join('acronym-watertight', obj_file_path)
        # print('Caution!!! We are currently handling non-simplified watertight meshes!!!')
        # each_h5_mesh = trimesh.load(watertight_obj_file_path, force='mesh', process=False)
        # # Visualize
        # save_obj_with_color(v=each_h5_mesh.vertices, f=each_h5_mesh.faces, c=None, file_name='debug_mesh_h5.obj')
        acronym_mesh_list.append({'obj_name': obj_name, 'obj_file_name': obj_file_name, 'obj_file_path': obj_file_path})
    return acronym_mesh_list


def load_dexycb_dataset(dexycb_models_path):
    # returns list of trimesh files for DexYCB objects
    dexycb_mesh_list = []
    for each_dexycb_model in os.listdir(dexycb_models_path):
        obj_name = each_dexycb_model[4:]
        obj_file_name = os.path.join(each_dexycb_model, 'textured')
        obj_file_path = os.path.join(each_dexycb_model, 'textured.obj')
        print(f'Processing {obj_file_name}....')
        dexycb_mesh_list.append({'obj_name': obj_name, 'obj_file_name': obj_file_name, 'obj_file_path': obj_file_path})
    return dexycb_mesh_list


# Initialze path for dataset
if target_dataset == 'Acronym':
    shapenetsem_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/ShapeNetSem/data/models-OBJ/models'
    acronym_watertight_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/acronym-watertight'
    simple_acronym_watertight_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/acronym-watertight_simplified'
elif target_dataset == 'ShapeNetSem':
    shapenetsem_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/ShapeNetSem/data/models-OBJ/models'
    shapenet_watertight_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/shapenet-watertight'
    simple_shapenet_watertight_path = '/mnt/disk1/danieljung0121/Hand2Object/models/Diffusion-SDF/datasets/shapenet-watertight_simplified'
elif target_dataset == 'DexYCB':
    dexycb_models_path = '/mnt/disk1/danieljung0121/Hand2Object/datasets/DexYCB/models'
elif target_dataset == 'ObMan':
    obman_models_path = '/mnt/disk1/danieljung0121/Hand2Object/datasets/DexYCB/models'
elif target_dataset == 'debug':
    pass
else:
    raise AttributeError('Not implemented!!!')


# Create watertight mesh from dataset
if create_watertight is True:
    if target_dataset == 'Acronym':
        print("Creating watertight dataset....")
        create_watertight_acronym(shapenetsem_path, acronym_watertight_path, simple_acronym_watertight_path)
    elif target_dataset == 'ShapeNetSem':
        print("Creating watertight dataset....")
        create_watertight_shapenet(shapenetsem_path, shapenet_watertight_path, simple_shapenet_watertight_path)
    elif target_dataset == 'DexYCB': # already watertight
        pass
    elif target_dataset == 'debug':
        pass
    else:
        raise AttributeError('Not implemented!!!')


# Choose target dataset
if target_dataset == 'Acronym':
    # Load Acronym dataset
    print("Loading Acronym dataset....")
    acronym_mesh_list = load_acronym_dataset(shapenetsem_path)
    target_mesh_list = acronym_mesh_list
elif target_dataset == 'ShapeNetSem':
    # Load Acronym dataset
    print("Loading Acronym dataset....")
    # shapenetsem_mesh_list = load_shapenetsem_dataset(shapenetsem_path)
    # target_mesh_list = shapenetsem_mesh_list
elif target_dataset == 'DexYCB':
    dexycb_mesh_list = load_dexycb_dataset(dexycb_models_path)
    target_mesh_list = dexycb_mesh_list
elif target_dataset == 'debug':
    mesh = trimesh.load(target_obj_path)
    mesh_name = 'chair'
    target_mesh_list = [{'obj_name': mesh_name, 'obj_file_name': target_obj_path.split('example/')[-1].split('.obj')[0], 'mesh': mesh}] #[mesh]
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


skip_rate = 4
skip_turn = 3 # Sever O running 0,1,2,3
skip_count = 0


for each_target_mesh_dict in target_mesh_list:
    if skip_count % skip_rate != skip_turn:
        skip_count += 1
        continue
    else:
        skip_count += 1
    # Extract sdf_data
    ext_sdf_start_time = time.time()
    print("Extracting SDFs from our own datasets....")
    obj_name = each_target_mesh_dict['obj_name']
    each_target_mesh_name = each_target_mesh_dict['obj_file_name']
    obj_file_path = each_target_mesh_dict['obj_file_path']
    if target_dataset == 'Acronym':
        mesh = trimesh.load(os.path.join('acronym-watertight', obj_file_path), force='mesh')
    elif target_dataset == 'DexYCB':
        mesh = trimesh.load(os.path.join('dexycb-models', obj_file_path), force='mesh')
    try:
        ext_points, ext_sdf = sample_sdf_near_surface(mesh, number_of_points=len(sdf_data)) # points: [596000, 3], sdf: [596000]
    except ValueError: # array with 0 samples
        continue
    ext_sdf_data = np.concatenate([ext_points, ext_sdf[:, None]], axis=-1)
    ext_voxels = mesh_to_voxels(mesh, 128, pad=False, sign_method='depth') ####### Changed from pad=True to pad=False
    ext_sdf_end_time = time.time()

    # Extract grid_gt
    ext_grid_start_time = time.time()
    grid_gt_query_points = grod_gt_coords
    mesh = scale_to_unit_sphere(mesh)
    ext_grid_gt_sdf = mesh_to_sdf(mesh, grid_gt_query_points, surface_point_method='scan', sign_method='depth', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
    ext_grid_gt = np.concatenate([grid_gt_query_points, ext_grid_gt_sdf[:, None]], axis=-1)
    filtered_grid_gt_query_points = grid_gt_query_points[ext_grid_gt_sdf < 0] # optional for visualization purpose
    ext_grid_end_time = time.time()


    # # Visualization
    # print("Visualizing SDFs from our own datasets....")
    # sdf_data_colors = np.zeros(ext_points.shape)
    # sdf_data_colors[ext_sdf < 0, 2] = 1
    # sdf_data_colors[ext_sdf > 0, 0] = 1
    # grid_gt_colors = np.zeros(grid_gt_query_points.shape)
    # grid_gt_colors[ext_grid_gt_sdf < 0, 2] = 1
    # grid_gt_colors[ext_grid_gt_sdf > 0, 0] = 1
    # save_obj_with_color(v=mesh.vertices, f=mesh.faces, c=None, file_name='debug_mesh.obj')
    # save_obj_with_color(v=ext_points, f=None, c=sdf_data_colors, file_name='debug_pc.obj')
    # save_obj_with_color(v=grod_gt_coords, f=None, c=None, file_name='debug_grid_gt.obj')
    # save_obj_with_color(v=grid_gt_query_points, f=None, c=grid_gt_colors, file_name='debug_ext_grid_gt.obj')
    # save_obj_with_color(v=filtered_grid_gt_query_points, f=None, c=None, file_name='debug_ext_grid_gt_filter.obj')


    # Save
    save_start_time = time.time()
    if not os.path.exists(os.path.join("full_diffsdf_data", target_dataset.lower(), obj_name, each_target_mesh_name)):
        os.makedirs(os.path.join("full_diffsdf_data", target_dataset.lower(), obj_name, each_target_mesh_name))
    if not os.path.exists(os.path.join("full_diffsdf_data", "grid_data", target_dataset.lower(), obj_name, each_target_mesh_name)):
        os.makedirs(os.path.join("full_diffsdf_data", "grid_data", target_dataset.lower(), obj_name, each_target_mesh_name))
    # np.savetxt(os.path.join("full_diffsdf_data", target_dataset.lower(), obj_name, each_target_mesh_name, "sdf_data.csv"), ext_sdf_data, delimiter=",")
    # np.savetxt(os.path.join("full_diffsdf_data", "grid_data", target_dataset.lower(), obj_name, each_target_mesh_name, "grid_gt.csv"), ext_grid_gt, delimiter=",")
    ext_sdf_data = pd.DataFrame(ext_sdf_data)
    ext_sdf_data.to_csv(os.path.join("full_diffsdf_data", target_dataset.lower(), obj_name, each_target_mesh_name, "sdf_data.csv"), header=False, index=False)
    ext_grid_gt = pd.DataFrame(ext_grid_gt)
    ext_grid_gt.to_csv(os.path.join("full_diffsdf_data", "grid_data", target_dataset.lower(), obj_name, each_target_mesh_name, "grid_gt.csv"), header=False, index=False)
    save_end_time = time.time()
    
    print('Extract sdf_data time is', str(ext_sdf_end_time - ext_sdf_start_time))
    print('Extract grid_gt time is', str(ext_grid_end_time - ext_grid_start_time))
    print('Save time is', str(save_end_time - save_start_time))