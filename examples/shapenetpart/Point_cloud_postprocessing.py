import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import open3d as o3d
import numpy as np
from collections import deque
import regiongrowing as reg



def pointscalarea(points,path=None): #计算叶面积

# 创建一个PointCloud对象
    pcd = o3d.geometry.PointCloud()
    # 将随机数转换成PointCloud点数据
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd.normals = o3d.utility.Vector3dVector(points[:,3:6])


    # path = "readyleaf.txt"
    # pcd = o3d.io.read_point_cloud(path,format='xyzn')

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd,
               o3d.utility.DoubleVector([radius, radius * 2]))
    area = mesh.get_surface_area()  # 计算表面积

#  将PointCloud点数据保存成ply文件，格式为assii文本格式
    if path != None:
        o3d.io.write_point_cloud(path, pcd, write_ascii=True)
    return area  #返回表面积

def pointcaldistance(points,path=None):  #计算茎长
    pcd = o3d.geometry.PointCloud()
    # 将随机数转换成PointCloud点数据
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])

    newpoint = np.asarray(pcd.points)
    sample_point = 2  # 想要采样的点的个数
    pcd_point = farthest_point_sample(newpoint, sample_point)  # FPS采样
    pcd_point = np.asarray(pcd_point)


    firstpoint = pcd_point[0,:]
    secondpoint =pcd_point[1,:]

    dis = distance_finder(firstpoint,secondpoint)

    #  将PointCloud点数据保存成ply文件，格式为assii文本格式
    if path != None:
        o3d.io.write_point_cloud(path, pcd, write_ascii=True)
    return dis  #返回两点间距离



def farthest_point_sample(point, npoint): #最远点采样
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def distance_finder(onepoint,twopoint) :  #欧氏距离
    [x1,y1,z1] = onepoint[:3]  # first coordinates
    [x2,y2,z2] = twopoint[:3]  # second coordinates

    return (((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))**(1/2)



def savepredictedply(points,path):  #保存点云

    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    # 将随机数转换成PointCloud点数据
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
    o3d.io.write_point_cloud(path, pcd)


def savepredictedtxt(nparr,path): #保存numpy以txt格式
    nparr = np.asarray(nparr)
    np.savetxt(path,nparr,fmt='%.6f')




def newcallengandwidth(leafpoints):   #新的计算叶长叶宽
    """
    首先，我们用主成分分析法(PCA) 计算了各个叶子点的第一、 第二和第三主成分向量。我们沿第一主成分向量找到两个端点。
    通过这两个端点之间的最短路径得到叶子的长度。
    其次，我们沿第一个主成分向量将叶点分为5个部分。
    然后在每个部分中，我们分别沿第二主成分向量和第三主成分向量找到两个端点。我们确定上述两个端点之间的最短路径中的最长路径为叶宽。
    """
    # 读取点云数据
    #pcd = o3d.io.read_point_cloud("leafcloud.ply")

    # 假设点云数据存储在numpy数组中，每一行是一个点的XYZ坐标
    points = np.asarray(leafpoints.points)
    #points = points[:, :3]
    # 计算PCA，并获取第一、第二、第三主成分向量
    pca = PCA(n_components=3)
    pca.fit(points)
    pc1, pc2, pc3 = pca.components_

    # 沿第一主成分向量找到两个端点，并计算叶子长度
    endpoints = np.array([points[np.argmin(points.dot(pc1))], points[np.argmax(points.dot(pc1))]])

    length = np.linalg.norm(endpoints[1] - endpoints[0])

    # 沿第一主成分向量将叶点分为5个部分
    z_min, z_max = np.argmin(points.dot(pc1)),np.argmax(points.dot(pc1))

    z_step = (z_max - z_min) // 5

    z_splits = np.arange(z_min, z_max, z_step)

    segments = [points[(points.dot(pc1) >= points.dot(pc1)[z1]) & (points.dot(pc1) < points.dot(pc1)[z2])] for z1, z2 in zip(list(z_splits[:-1]), list(z_splits[1:]))]

    # 在每个部分中沿第二和第三主成分向量找到两个端点，并计算叶宽
    widths = []
    for segment in segments:
        if len(segment) == 0:
            widths.append(0)
            continue
        endpoints2 = np.array([segment[np.argmin(segment.dot(pc2))], segment[np.argmax(segment.dot(pc2))]])
        endpoints3 = np.array([segment[np.argmin(segment.dot(pc3))], segment[np.argmax(segment.dot(pc3))]])
        pairs = [(endpoints2[0], endpoints3[0]), (endpoints2[0], endpoints3[1]), (endpoints2[1], endpoints3[0]), (endpoints2[1], endpoints3[1])]
        distances = [np.linalg.norm(pair[0] - pair[1]) for pair in pairs]
        min_distance = np.min(distances)
        widths.append(min_distance)
    if len(widths) == 0:
        width = 0
    else:
        width = max(widths)
    # 输出结果
    #print(f"Leaf length: {length}")
   # print(f"Leaf widths: {width}")
    return length,width


def calculatestem(points):  #计算茎直径
#只需要坐标

    """
    首先，我们沿Z轴将茎点云分成4个均匀的部分。
    然后，我们使用最小二乘法在茎点的最小Z值部分上拟合出一条直线段。
    最后，我们计算从这些茎点到这条直线的投影距离，并选择这些距离中值的两倍作为茎直径。
    """
    # 读取点云数据
    #pcd = o3d.io.read_point_cloud("leafcloud.ply")

    # 假设点云数据存储在numpy数组中，每一行是一个点的XYZ坐标
    #points = np.array(points)
    points = np.asarray(points)
    points = points[:, :3]
    # 沿Z轴将点云分成4个部分
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_step = (z_max - z_min) / 4
    z_splits = np.arange(z_min + z_step, z_max, z_step)
    segments = [points[(points[:, 2] >= z1) & (points[:, 2] < z2)] for z1, z2 in
                zip([z_min] + list(z_splits), z_splits + [z_max])]

    # 在最小Z值部分上拟合直线段
    min_z_segment = segments[0]
    model = LinearRegression().fit(min_z_segment[:, :2], min_z_segment[:, 2])
    z_pred = model.predict(min_z_segment[:, :2])
    residuals = min_z_segment[:, 2] - z_pred

    # 计算茎直径
    diameter = 2 * np.median(np.abs(residuals))
    return diameter

def leafinstancesegment(points):  #叶片实例分割

    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
    rg = reg.RegionGrowing(pcd,
                           min_pts_per_cluster=30,  # 每个聚类的最小点数
                           max_pts_per_cluster=2048,  # 每个聚类的最大点数
                           neighbour_number=10,  # 邻域搜索点数，将其减少可以限制邻域搜索范围，从而使得聚类数量减少
                           theta_threshold=100,  # 法向量夹角阈值，将其增加，这样可以使得较不规则的区域不容易被聚类到一起，从而使聚类数量减少。
                           curvature_threshold=0.01)  # 曲率阈值，将其增加，这样可以使得较不规则的区域不容易被聚类到一起，从而使聚类数量减少。
    indices = rg.extract()
    # ---------------------------聚类结果分类保存----------------------------------
    segment = []  # 存储分割结果的容器
    leafnum = len(indices)  # 叶片数量
    for i in range(len(indices)):
        ind = indices[i]
        clusters_cloud = pcd.select_by_index(ind)
        num_points = len(clusters_cloud.points)  # 统计聚类中的点数
        if num_points <= 70:  #最小成叶片的点数限制
            leafnum = leafnum - 1
        else:
            segment.append(clusters_cloud)

    return segment, leafnum  #返回叶片实例数组和叶片数量

    # pathfile = "D:/pythonproject/Pointnext/examples/shapenetpart/predictedpoints/leaves/"
    # for index, item in enumerate(segment):
    #     path = pathfile + str(index + 1) + "_leaf.ply"
    #     o3d.io.write_point_cloud(path, item)

