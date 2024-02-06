import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import csv
import json
import io
from PIL import Image
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels
import pyrender


sdf_data_path = 'diffsdf_data/acronym/Couch/37cfcafe606611d81246538126da07a8/sdf_data.csv'
grid_gt_path = 'diffsdf_data/grid_data/acronym/Couch/37cfcafe606611d81246538126da07a8/grid_gt.csv'
splits_path = 'diffsdf_data/splits/couch_all.json'

target_obj_path = 'example/chair.obj'
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
print("Extracting SDFs from our own datasets....")
mesh = trimesh.load(target_obj_path)
ext_points, ext_sdf = sample_sdf_near_surface(mesh, number_of_points=len(sdf_data)) # points: [596000, 3], sdf: [596000]
ext_sdf_data = np.concatenate([ext_points, ext_sdf[:, None]], axis=-1)

ext_voxels = mesh_to_voxels(mesh, 128, pad=False, sign_method='depth') ####### Changed from pad=True to pad=False


# Visualization
print("Visualizing SDFs from our own datasets....")
colors = np.zeros(ext_points.shape)
colors[ext_sdf < 0, 2] = 1
colors[ext_sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(ext_points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)



# render scene
# Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
s = np.sqrt(2)/2
camera_pose = np.array([
       [0.0, -s,   s,   1.5],
       [1.0,  0.0, 0.0, 0.0],
       [0.0,  s,   s,   1.5],
       [0.0,  0.0, 0.0, 1.0],
    ])
scene.add(camera, pose=camera_pose)

# Set up the light -- a single spot light in the same spot as the camera
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0)
scene.add(light, pose=camera_pose)

# Render the scene
r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)
cv2.imwrite('debug.png', color)

save_obj_with_color(v=mesh.vertices, f=mesh.faces, c=None, file_name='debug_mesh.obj')
save_obj_with_color(v=ext_points, f=None, c=colors, file_name='debug_pc.obj')
import pdb; pdb.set_trace()