import os
import open3d as o3d
import numpy as np
from openpoints.models.layers import fps
import torch




def point_cloud_normalize(cloud):
    """
    对点云数据进行归一化
    :param cloud: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    centroid = cloud.get_center()  # 计算点云质心
    points = np.asarray(cloud.points)
    points = points - centroid     # 去质心
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))  # 计算点云中的点与坐标原点的最大距离
    points = points / m  # 对点云进行缩放
    normalize_cloud = o3d.geometry.PointCloud()  # 使用numpy生成点云
    normalize_cloud.points = o3d.utility.Vector3dVector(points)
   # normalize_cloud.colors = cloud.colors  # 获取投影前对应的颜色赋值给投影后的点
    return normalize_cloud

fileoutputarr = []
path = "D:/实验室组会/论文/基于高通量表型的棉花苗期胁迫分析/原稿/PP后修改/大豆/archive/"
savepath = "D:/实验室组会/论文/基于高通量表型的棉花苗期胁迫分析/原稿/PP后修改/大豆/outputarchive/"
dirlist = os.listdir(path)
filepatharr = []
for dir in dirlist:
    fileoutputarr.append(dir)
    filepatharr.append(path+dir + "/Annotations/")

i = 0
for filepath in filepatharr:  # 0为叶子 1为主干 2为分支
    plyarr = os.listdir(filepath)   #为一个点云
    plydata = None
    flag = 1
    for plyfile in plyarr:
        plypath = filepath + plyfile
        tempdata = np.loadtxt(plypath)
        tempdata = tempdata[:,:3]
        if "leaf" in plypath:

            tempdata = np.concatenate((tempdata,np.asarray(([0]*len(tempdata))).reshape(len(tempdata),1)),axis=1)
        elif "mainstem" in plypath:
            tempdata = np.concatenate((tempdata,np.asarray(([1]*len(tempdata))).reshape(len(tempdata),1)),axis=1)
        elif "stem" in plypath:
            tempdata = np.concatenate((tempdata,np.asarray(([2]*len(tempdata))).reshape(len(tempdata),1)),axis=1)

        if flag:
            plydata = tempdata
            flag = 0

        else:
            plydata = np.concatenate((plydata, tempdata), axis=0)

    plydata = torch.from_numpy(plydata).to(
        torch.float32).cuda().unsqueeze(0)
    plydata = fps(plydata, 32768).cpu().numpy()[0]

    #################
    temppcd = o3d.geometry.PointCloud()
    temppcd.points = o3d.utility.Vector3dVector(plydata[:,:3])
    temppcd = point_cloud_normalize(temppcd)
    temppcd.estimate_normals(  # 计算法向量
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    npx = np.asarray(temppcd.normals)
    px = np.asarray(temppcd.points)
    data = np.concatenate((px,npx),axis=1)
    plydata = np.concatenate((data, plydata[:,-1:]), axis=1)
    ###################

    np.savetxt(savepath+fileoutputarr[i]+".txt",plydata,fmt='%.6f')
    i = i + 1




import numpy as np
import open3d as o3d












