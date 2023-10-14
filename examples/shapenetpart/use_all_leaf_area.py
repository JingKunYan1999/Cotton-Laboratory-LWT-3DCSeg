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
for i in range(len(indices)):
    ind = indices[i]
    clusters_cloud = pcd.select_by_index(ind)
  #  num_points = len(clusters_cloud.points)  # 统计聚类中的点数
   # total_points += num_points  # 累加聚类中的点数
 #   r_color = np.random.uniform(0, 1, (1, 3))  # 分类点云随机赋色
   # clusters_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
    segment.append(clusters_cloud)

# -----------------------------错误提示：判断总点数是否小于点云文件中点的总数的99.9%------------------------------------
#total_points_file = len(pcd.points)
# if total_points < 0.999*total_points_file:
#     print("参数不合规，聚类点数过少。")
#     exit()

# -----------------------------数量可视化------------------------------------
print("叶片个数为", len(indices))
# print("总点数为", total_points)

# -----------------------------输出主茎的类------------------------------------
# 将点云转换为numpy数组
points = np.asarray(pcd.points)

# 计算最小高度和最大高度
min_height = np.min(points[:, 2])

# 将Z方向上的最小高度增加20，得到新的高度值
new_height = min_height + 20

# 选取高度在 new_height 正负 1 单位内的点
mask = np.logical_and(points[:, 2] >= new_height - 1, points[:, 2] <= new_height + 1)
selected_points = points[mask]

# 输出高度在 new_height 正负 1 单位内的点所在的聚类编号

for i in range(len(segment)):
    cluster_points = np.asarray(segment[i].points)
    mask = np.logical_and(cluster_points[:, 2] >= new_height - 1, cluster_points[:, 2] <= new_height + 1)
    if mask.any():
        selected_cluster_idx = i
        break

# 随机选择该聚类中的一个点
selected_cluster_points = np.asarray(segment[selected_cluster_idx].points)
random_point_idx = np.random.choice(np.where(mask)[0])
selected_point = selected_cluster_points[random_point_idx]

# 输出该点所在的聚类编号
# print("随机选取的点所在的聚类编号为：", selected_cluster_idx)

# -----------------------------计算植株表面积------------------------------------
mesh = o3d.io.read_triangle_mesh(plyFilePath)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0.7, 0])
area = mesh.get_surface_area()  # 计算表面积
# print("植株表面积为：", area)

# -----------------------------计算主茎所在聚类的表面积------------------------------------
# 将 selected_cluster_points 转换为 PointCloud 对象
selected_cluster_pcd = o3d.geometry.PointCloud()
selected_cluster_pcd.points = o3d.utility.Vector3dVector(selected_cluster_points)

# 创建 alpha shape，并计算表面积
alpha = 2.0
cluster_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(selected_cluster_pcd, alpha=alpha)
surface_area = cluster_mesh.get_surface_area()
# print("主茎所在聚类的表面积为:", surface_area)


print("总叶面积为:", area - surface_area)
# -----------------------------结果可视化------------------------------------
o3d.visualization.draw_geometries(segment, window_name="区域生长分割",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)



