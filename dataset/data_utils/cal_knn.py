from .o3d import kdtree as o3d_kdtree
from sklearn.neighbors import KDTree as sk_kdtree
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import numpy as np
import pickle
import os
import logging

'''
FAISS_INSTALLED = False
try:
    faiss = import_module('faiss')
    FAISS_INSTALLED = True
except Exception as e:
    print(e)
    print('Cannot import faiss for GPU nearest neighbout search, use Open3d instead.')
    
SKLEARN_INSTALLED = False
try:
    _neighbours = import_module("sklearn.neighbors")
    sk_kdtree = getattr(_neighbours, 'KDTree')
    SKLEARN_INSTALLED = True
except Exception as e:
    print(e)
    print('Cannot import sklearn for nearest neighbout search, use Open3d instead.')

'''

class _NearestNeighbors(object):
    def __init__(self, set_k=None, **kwargs):
        self.model = None
        self.set_k = set_k

    def train(self, data):
        pass
    
    @staticmethod
    def save(filename):
        return
    
    @staticmethod
    def load(filename):
        return 

    def search(self, data, k, return_distance=True):
        if self.set_k is not None:
            assert self.set_k == k, \
                'K not match to setting {}'.format(self.set_k)
        D, I = None, None
        return D, I


class Open3dNN(_NearestNeighbors):
    def __init__(self, set_k=None, **kwargs):
        super(Open3dNN, self).__init__(set_k, **kwargs)
        self.model = None
    def train(self, data):
        assert data.shape[1] == 3, 'Must be shape [?, 3] for point data'
        self.model = o3d_kdtree(data)

    def search(self, data, k, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if data.shape[0] == 1:
            [__, I, _] = self.model.search_knn_vector_3d(data[0], k)
        else:
            I = np.zeros((data.shape[0], k), dtype=np.int)
            with ThreadPoolExecutor(256) as executor:
                for i in range(I.shape[0]):
                    executor.submit(self._search_multiple, (self.model, I, data, k, i,))
        return None, I
    # modified need !
    def search_radius(self, data, radius, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[0]== 1:
            return self.model.query_radius(data, r=radius)[0]
        else:
            return self.model.query_radius(data, r=radius)
        
    def save(self, filename):
        assert self.model is not None, "Model have not been trained"
        pickle.dump(self.model, open(filename, 'wb'))
    
    @staticmethod
    def load(filename):
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        init_nn = SkNN()
        init_nn.model = pickle.load(open(filename, 'rb')) 
        return init_nn

    @staticmethod
    def _search_multiple(knn_searcher, I, data, k, i):
            [__, I_, _] = knn_searcher.search_knn_vector_3d(data[i, :], k)
            I[i, :] = np.asarray(I_)


class SkNN(_NearestNeighbors):
    def __init__(self, set_k=None, **kwargs):
        super(SkNN, self).__init__(set_k, **kwargs)
        self.model = None
        if 'leaf_size' in kwargs.keys():
            self.leaf_size = kwargs['leaf_size']
        else:
            self.leaf_size = 40
            
    def save(self, filename):
        assert self.model is not None, "Model have not been trained"
        pickle.dump(self.model, open(filename, 'wb'))
    
    @staticmethod
    def load(filename):
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        init_nn = SkNN()
        init_nn.model = pickle.load(open(filename, 'rb')) 
        return init_nn

    def train(self, data):
        self.model = sk_kdtree(data, leaf_size=self.leaf_size)

    def search(self, data, k, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[0]== 1:
            return self.model.query(data, k)[0]
        else:
            return self.model.query(data, k)
            
    def search_radius(self, data, radius, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[0]== 1:
            return self.model.query_radius(data, r=radius)[0]
        else:
            return self.model.query_radius(data, r=radius)
'''
class FaissNN(_NearestNeighbors):
    #GPU KNN Search for large scale
    def __init__(self, set_k=None, **kwargs):
        super(FaissNN, self).__init__(set_k, **kwargs)
        self.IVF_number = 32786
        self.GPU_id = None
        if isinstance(kwargs, dict):
            if 'IVF_number' in kwargs: self.IVF_number = kwargs['IVF_number']
            if 'GPU_id' in kwargs: self.GPU_id = kwargs['GPU_id']
        self.model = None
        self.dimension = None

    def train(self, data):
        d = data.shape[1]
        data = data.astype(np.float32)
        self.model = faiss.index_factory(int(d), 'IVF{}_HNSW32,Flat'.format(self.IVF_number)) #_HNSW32
        if self.GPU_id is not None and isinstance(self.GPU_id, int):
            res = faiss.StandardGpuResources()
            self.model = faiss.index_cpu_to_gpu(res, self.GPU_id, self.model)
        elif isinstance(self.GPU_id, list):
            #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in self.GPU_id])
            self.model = faiss.index_cpu_to_all_gpus(self.model)
        else:
            self.model = faiss.index_cpu_to_all_gpus(self.model)
        self.model.train(data)
        self.model.add(data)
        self.model.nprobe = d ** 2

    def search(self, data, k, return_distance=True):
        data = data.astype(np.float32)
        assert self.model is not None, "Model have not been trained"
        #assert self.model.is_trained, "Model not trained."
        D, I = self.model.search(data, k)
        if return_distance: D = None
        return D, I
'''
if __name__ == "__main__":
    import open3d
    import sys
    import time
    for r in ['PGP', 'FOE', 'FASS', 'UCC', 'YIH', 'RA']:
        pts = np.asarray(open3d.io.read_point_cloud("/home/lixk/liaoyq/Campus3D_v2/data/campus3D/{}/{}.pcd".format(r,r)).points)
        nn = SkNN()
        tic = time.time()
        nn.train(pts[:, :-1])
        nn.save("/home/lixk/liaoyq/Campus3D_v2/data/campus3D/{}/{}.pickle".format(r,r))
        print(time.time()-tic, r)
    sys.exit(0)
    
    
    
    
    import sys
    import time
    import os

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    nb = 10**5
    nq = 10**5
    np.random.seed(1)
    datab = np.random.rand(nb, 3).astype('float32')
    dataq = np.random.rand(nq, 3).astype('float32')

    tic = time.time()
    nn = SkNN(set_k=3)
    nn.train(datab)
    print(time.time() - tic)
    tic = time.time()
    D, I = nn.search(dataq, 3)
    print(time.time() - tic)




