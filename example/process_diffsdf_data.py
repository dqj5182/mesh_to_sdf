import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import csv
import json
import trimesh
import numpy as np

from mesh_to_sdf import sample_sdf_near_surface


sdf_data_path = 'diffsdf_data/acronym/Couch/37cfcafe606611d81246538126da07a8/sdf_data.csv'
grid_gt_path = 'diffsdf_data/grid_data/acronym/Couch/37cfcafe606611d81246538126da07a8/grid_gt.csv'
splits_path = 'diffsdf_data/splits/couch_all.json'

target_obj_path = 'example/chair.obj'

print("Loading sample files from Diffusion SDF....")
with open(sdf_data_path, newline='') as f: # sdf_data: Points on and near the mesh
    reader = csv.reader(f)
    sdf_data = np.array(list(reader)) # [596000, 4]

with open(grid_gt_path, newline='') as f: # grid_gt: SDF values for points distributed on a grid, and are intended to sample empty areas
    reader = csv.reader(f)
    grid_gt = np.array(list(reader)) # [468000, 4]

with open(splits_path) as f:
    splits = json.load(f)


# Extract sdf_data
mesh = trimesh.load(target_obj_path)
ext_points, ext_sdf = sample_sdf_near_surface(mesh, number_of_points=len(sdf_data)) # points: [596000, 3], sdf: [596000]
ext_sdf_data = np.concatenate([ext_points, ext_sdf[:, None]], axis=-1)

import pdb; pdb.set_trace()