from ..data_utils import cal_knn
import numpy as np
import os


class KnnQuery(object):
    def __init__(self, points, knn_module, set_k):
        knn_module = getattr(cal_knn, knn_module)  # importlib.import_module('.'.join(['cal_knn', self.knn_module]))
        self.knn_searcher = knn_module(set_k=set_k)
        self.knn_searcher.train(points)

    def search(self, point, k):
        if len(point.shape) < 2:
            point = np.expand_dims(point, axis=0)
        return self.knn_searcher.search(point, k)


class BlockQuery(object):
    def __init__(self, inputs, knn_module, block_size):
        #use 2D data
        knn_module = getattr(cal_knn, knn_module)
        if isinstance(inputs, str):
            #assert os.path.isfile(inputs), '{} is not a valid file'.format(inputs)
            self.knn_model = knn_module.load(inputs)
        else:
            self.knn_model = knn_module()
            self.knn_model.train(inputs)
        self.block_size = np.array(block_size)
        self._radius = np.sqrt(np.sum(self.block_size**2))
        #logger.info('Getting search dictionary')

    def search_candidates(self, center_pt, *args):
        center_pt = center_pt[:-1] if len(center_pt.shape) == 1 else center_pt[:, :-1]
        return self.knn_model.search_radius(center_pt, radius=self._radius)

    def search(self, center, points):
        assert len(points.shape) == 2, 'Points must be in shape N x dim'
        points = points[:, :-1]
        candidate_index = self.search_candidates(center)
        if True:
            candidate_points = points[candidate_index]
            mask_x = np.logical_and(candidate_points[:, 0] <= center[0] + self.block_size[0] / 2,
                                    candidate_points[:, 0] >= center[0] - self.block_size[0] / 2)
            mask_y = np.logical_and(candidate_points[:, 1] <= center[1] + self.block_size[1] / 2,
                                    candidate_points[:, 1] >= center[1] - self.block_size[1] / 2)
            return candidate_index[np.logical_and(mask_x, mask_y)]
        
    def save(self, filename ):
        self.knn_model.save(filename)
        


if __name__ == "__main__":
    import open3d
    import sys
    import time
    pnum = 1000000
    s = np.array([1,1,1])
    knn_module = 'SkNN'
    #pts = gen_point_cloud(high=1000, low=1, center_num=20, size=pnum, scale=0.3)
    pts = np.asarray(open3d.io.read_point_cloud("/home/lixk/liaoyq/Campus3D_v2/data/campus3D/PGP/PGP.pcd").points)
    sp_i = np.random.randint(1, pts.shape[0], 1000)
    #knn = KnnQuery(pts, knn_module, 2048)
    q = BlockQuery(pts[:, :-1])
    tic = time.time()
    for i in range(100):
        _ = q.search(pts[sp_i[i]], 2048)
    toc = time.time()
    print((toc - tic) / 100)
