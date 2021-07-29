import open3d
import h5py
import numpy as np
from data_utils.o3d import create_point_cloud, voxel_sampling

#convert dense point cloud in h5 format into sparse point cloud with pcd and npy format.

def read_h5(h5_file, region):
    with h5py.File(h5_file) as hf:
        d = hf[region]
        coordinates = np.asarray([d['x'],d['y'],d['z']]).transpose()
        colors = np.asarray([d['r'],d['g'],d['b']]).transpose()
        labels = np.asarray([d['l' + str(n)] for n in range(23)]).transpose()
    return coordinates, colors, labels

def get_trace_index(inds):
    row_ind = np.argmax(inds>=0, axis=1)
    return inds[np.arange(len(row_ind)), row_ind]
        
def preprocess(h5_file, region, pcd_file_name, label_file_name):
    coordinates, colors, labels = read_h5(h5_file, region)
    pcd = create_point_cloud(points=coordinates, colors=colors)
    down_pcd, inds = voxel_sampling(points, voxel_size, return_index=True)
    down_label = labels[get_trace_index(inds), :]
    open3d.io.write_point_cloud(pcd_file_name, pcd)
    np.save(label_file_name, down_label)
