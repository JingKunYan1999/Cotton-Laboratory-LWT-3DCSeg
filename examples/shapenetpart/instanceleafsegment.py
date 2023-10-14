import open3d as o3d
import numpy as np
import regiongrowing as reg

# ------------------------------读取点云---------------------------------------
plyFilePath = 'D:/pythonproject/Pointnext/examples/shapenetpart/predictedpoints/3_leaf.ply'
pcd = o3d.io.read_point_cloud(plyFilePath)
# ------------------------------区域生长---------------------------------------
rg = reg.RegionGrowing(pcd,
                       min_pts_per_cluster=10,     # 每个聚类的最小点数
                       max_pts_per_cluster=1000000,  # 每个聚类的最大点数
                       neighbour_number=10,         # 邻域搜索点数，将其减少可以限制邻域搜索范围，从而使得聚类数量减少
                       theta_threshold=100,          # 法向量夹角阈值，将其增加，这样可以使得较不规则的区域不容易被聚类到一起，从而使聚类数量减少。
                       curvature_threshold=0.01)    # 曲率阈值，将其增加，这样可以使得较不规则的区域不容易被聚类到一起，从而使聚类数量减少。
indices = rg.extract()

# ---------------------------聚类结果分类保存----------------------------------
total_points = 0  # 总点数
segment = []  # 存储分割结果的容器
leafnum = len(indices) #叶片数量
for i in range(len(indices)):
    ind = indices[i]
    clusters_cloud = pcd.select_by_index(ind)
    num_points = len(clusters_cloud.points)  # 统计聚类中的点数
    if num_points <= 10:
        leafnum = leafnum -1
    else:
        segment.append(clusters_cloud)


pathfile = "D:/pythonproject/Pointnext/examples/shapenetpart/predictedpoints/leaves/"
for index,item in enumerate(segment):
    path = pathfile + str(index+1) + "_leaf.ply"
    o3d.io.write_point_cloud(path, item)
