import open3d as o3d
import numpy as np
from collections import deque


class RegionGrowing:

    # 构造函数
    def __init__(self, cloud,
                 min_pts_per_cluster=1,             # 每个聚类的最小点数
                 max_pts_per_cluster=np.inf,        # 每个聚类的最大点数
                 theta_threshold=30,                # 法向量夹角阈值
                 curvature_threshold=0.05,          # 曲率阈值
                 neighbour_number=30,               # 邻域搜索点数
                 point_neighbours=[],               # 近邻点集合
                 point_labels=[],                   # 点标签
                 num_pts_in_segment=[],             # 分类标签
                 clusters=[],                       # 聚类容器
                 number_of_segments=0):             # 聚类个数

        self.cure = None                                 # 存储每个点曲率的容器
        self.pcd = cloud                                 # 输入点云
        self.min_pts_per_cluster = min_pts_per_cluster
        self.max_pts_per_cluster = max_pts_per_cluster
        self.theta_threshold = np.deg2rad(theta_threshold)
        self.curvature_threshold = curvature_threshold
        self.neighbour_number = neighbour_number

        self.point_neighbours = point_neighbours
        # print("打印self.point_neighbours：", len(self.point_neighbours))

        self.point_labels = point_labels
        # print("打印self.point_labels：", len(self.point_neighbours))

        self.num_pts_in_segment = num_pts_in_segment
        self.num_pts_in_segment.clear()
        # print("打印self.num_pts_in_segment：", len(self.num_pts_in_segment))

        self.clusters = clusters
        self.clusters.clear()
        # print("打印self.clusters：", len(clusters))

        self.number_of_segments = number_of_segments
        # print("打印self.number_of_segments：", self.number_of_segments)

    # -------------------------------------参数准备--------------------------------------
    def prepare_for_segment(self):
        points = np.asarray(self.pcd.points)     # 点坐标
        normals = np.asarray(self.pcd.normals)   # 法向量
        # 判断点云是否为空
        if not points.shape[0]:
            return False
        # 判断是否有近邻点
        if self.neighbour_number == 0:
            return False
        # 点云需要包含法向量信息
        if points.shape[0] != normals.shape[0]:
            self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))

        return True

    # ------------------------------------近邻点搜索-------------------------------------
    def find_neighbour_points(self):
        number = len(self.pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.point_neighbours = np.zeros((number, self.neighbour_number))
        for ik in range(number):
            [_, idx, _] = kdtree.search_knn_vector_3d(self.pcd.points[ik], self.neighbour_number)  # K近邻搜索
            self.point_neighbours[ik, :] = idx

    # -----------------------------------判意点所属分类-----------------------------------
    def validate_points(self, point, nebor):
        is_seed = True
        cosine_threshold = np.cos(self.theta_threshold)  # 法向量夹角（平滑）阈值

        curr_seed_normal = self.pcd.normals[point]       # 当前种子点的法向量
        seed_nebor_normal = self.pcd.normals[nebor]      # 种子点邻域点的法向量
        dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))
        # 如果小于平滑阈值
        if dot_normal < cosine_threshold:
            return False, is_seed
        # 如果小于曲率阈值
        if self.cure[nebor] > self.curvature_threshold:
            is_seed = False

        return True, is_seed

    # ----------------------------------对点附上分类标签----------------------------------
    def label_for_points(self, initial_seed, segment_number):
        seeds = deque([initial_seed])
        self.point_labels[initial_seed] = segment_number
        num_pts_in_segment = 1

        while len(seeds):
            curr_seed = seeds[0]
            seeds.popleft()
            i_nebor = 0
            while i_nebor < self.neighbour_number and i_nebor < len(self.point_neighbours[curr_seed]):
                index = int(self.point_neighbours[curr_seed, i_nebor])
                if self.point_labels[index] != -1:
                    i_nebor += 1
                    continue

                belongs_to_segment, is_seed = self.validate_points(curr_seed, index)
                if not belongs_to_segment:
                    i_nebor += 1
                    continue

                self.point_labels[index] = segment_number
                num_pts_in_segment += 1

                if is_seed:
                    seeds.append(index)

                i_nebor += 1

        return num_pts_in_segment

    # ------------------------------------区域生长过程------------------------------------
    def region_growing_process(self):
        num_of_pts = len(self.pcd.points)         # 点云点的个数
        self.point_labels = -np.ones(num_of_pts)  # 初始化点标签
        self.pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))
        cov_mat = self.pcd.covariances            # 获取每个点的协方差矩阵
        self.cure = np.zeros(num_of_pts)          # 初始化存储每个点曲率的容器
        # 计算每个点的曲率
        for i_n in range(num_of_pts):
            eignvalue, _ = np.linalg.eig(cov_mat[i_n])  # SVD分解求特征值
            idx = eignvalue.argsort()[::-1]
            eignvalue = eignvalue[idx]
            self.cure[i_n] = eignvalue[2] / (eignvalue[0] + eignvalue[1] + eignvalue[2])

        point_curvature_index = np.zeros((num_of_pts, 2))
        for i_cu in range(num_of_pts):
            point_curvature_index[i_cu, 0] = self.cure[i_cu]
            point_curvature_index[i_cu, 1] = i_cu

        # 按照曲率大小进行排序
        temp_cure = np.argsort(point_curvature_index[:, 0])
        point_curvature_index = point_curvature_index[temp_cure, :]

        seed_counter = 0
        seed = int(point_curvature_index[seed_counter, 1])  # 选取曲率最小值点

        segmented_pts_num = 0
        number_of_segments = 0

        while segmented_pts_num < num_of_pts:
            pts_in_segment = self.label_for_points(seed, number_of_segments)  # 根据种子点进行分类
            segmented_pts_num += pts_in_segment
            self.num_pts_in_segment.append(pts_in_segment)
            number_of_segments += 1

            # 寻找下一个种子
            for i_seed in range(seed_counter + 1, num_of_pts):
                index = int(point_curvature_index[i_seed, 1])
                if self.point_labels[index] == -1:
                    seed = index
                    seed_counter = i_seed
                    break

    # ----------------------------------根据标签进行分类-----------------------------------
    def region_growing_clusters(self):
        number_of_segments = len(self.num_pts_in_segment)
        number_of_points = np.asarray(self.pcd.points).shape[0]

        # 初始化聚类数组
        for i in range(number_of_segments):
            tmp_init = list(np.zeros(self.num_pts_in_segment[i]))
            self.clusters.append(tmp_init)

        counter = list(np.zeros(number_of_segments))
        for i_point in range(number_of_points):
            segment_index = int(self.point_labels[i_point])
            if segment_index != -1:
                point_index = int(counter[segment_index])
                self.clusters[segment_index][point_index] = i_point
                counter[segment_index] = point_index + 1

        self.number_of_segments = number_of_segments

    # ----------------------------------执行区域生长算法-----------------------------------
    def extract(self):
        if not self.prepare_for_segment():
            print("区域生长算法预处理失败！")
            return

        self.find_neighbour_points()
        self.region_growing_process()
        self.region_growing_clusters()

        # 根据设置的最大最小点数筛选符合阈值的分类
        all_cluster = []
        # print("叶面积列表内元素数量：", len(all_cluster))
        for i in range(len(self.clusters)):
            if self.min_pts_per_cluster <= len(self.clusters[i]) <= self.max_pts_per_cluster:
                all_cluster.append(self.clusters[i])
            else:
                self.point_labels[self.clusters[i]] = -1


        self.clusters = all_cluster
        return all_cluster

