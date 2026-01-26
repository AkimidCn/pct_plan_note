#!/usr/bin/python3
import os
import sys
import time
import pickle
import numpy as np
import open3d as o3d
  
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from tomogram import Tomogram

sys.path.append('../')
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class Tomography(object):
    def __init__(self, cfg, scene_cfg):
        self.export_dir = rsg_root + cfg.map.export_dir
        self.pcd_file = scene_cfg.pcd.file_name
        self.resolution = scene_cfg.map.resolution
        self.ground_h = scene_cfg.map.ground_h   # 最低点高度?
        self.slice_dh = scene_cfg.map.slice_dh   # 对应 d_s (切片间距)

        self.center = np.zeros(2, dtype=np.float32)
        self.tomogram = Tomogram(scene_cfg)
        points = self.loadPCD(self.pcd_file)     # 读取点云

        # Process
        self.process(points)

    def initROS(self):  # 创建点云发布器
        self.map_frame = cfg.ros.map_frame

        pointcloud_topic = cfg.ros.pointcloud_topic
        self.pointcloud_pub = rospy.Publisher(pointcloud_topic, PointCloud2, latch=True, queue_size=1)

        self.layer_G_pub_list = []
        self.layer_C_pub_list = []
        layer_G_topic = cfg.ros.layer_G_topic
        layer_C_topic = cfg.ros.layer_C_topic
        for i in range(self.n_slice):
            layer_G_pub = rospy.Publisher(layer_G_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_G_pub_list.append(layer_G_pub)
            layer_C_pub = rospy.Publisher(layer_C_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_C_pub_list.append(layer_C_pub)

        tomogram_topic = cfg.ros.tomogram_topic
        self.tomogram_pub = rospy.Publisher(tomogram_topic, PointCloud2, latch=True, queue_size=1)

    def loadPCD(self, pcd_file):
        pcd = o3d.io.read_point_cloud(rsg_root + "/rsc/pcd/" + pcd_file)
        points = np.asarray(pcd.points).astype(np.float32)   # 转换为 numpy 格式,points.shape=(N,3)
        rospy.loginfo("PCD points: %d", points.shape[0])   # 点数量

        if points.shape[1] > 3:
            points = points[:, :3]
        self.points_max = np.max(points, axis=0)   # 按行找出每列最大值,即 x,y,z 最大值,
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h    # -1为最后一个值的索引,设置最低点高度(z)
        
        # 计算在x,y方向上的维度（格子数量）,加4是为了边界扩展,保证安全
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4    # ceil是向上取整
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))  # 计算切片层数
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh    # 首层切片高度
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        rospy.loginfo("Map center: [%.2f, %.2f]", self.center[0], self.center[1])
        rospy.loginfo("Dim_x: %d", self.map_dim_x)
        rospy.loginfo("Dim_y: %d", self.map_dim_y)
        rospy.loginfo("Num slices init: %d", n_slice_init)

        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        return points
        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 10

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            
            # 求各层的代价、梯度等  layers_g形状为 [n_slice, n_row, n_col]
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        rospy.loginfo("Num slices simp: %d", layers_g.shape[0])
        rospy.loginfo("Num repeats (for benchmarking only): %d", n_repeat)
        rospy.loginfo(" -- avg t_map  (ms): %f", t_map / n_repeat)
        rospy.loginfo(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        rospy.loginfo(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        rospy.loginfo(" -- avg t_all  (ms): %f", t_all / n_repeat)

        self.n_slice = layers_g.shape[0]

        # 保存文件
        map_file = os.path.splitext(self.pcd_file)[0]  # 分割文件名的扩展名，取分割后的第一部分（不含扩展名），将data/scene.pcd变为"data/scene"
        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file) # 导出断层图数据

        self.initROS()  # 创建点云发布器
        self.publishPoints(points) # 发布原始点云
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)  # 根据layers_t代价来决定颜色强度
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)

    def exportTomogram(self, tomogram, map_file):        
        data_dict = {
            'data': tomogram.astype(np.float16),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
        }
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Tomogram exported: %s", file_name)

    def publishPoints(self, points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        point_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(point_msg)

    def publishLayers(self, pub_list, layers, color=None):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        layer_points = self.VISPROTO_P.copy()  # 预定义的点云模板  。copy是深拷贝为了不改变原数据
        layer_points[:, :2] += self.center  # center是点云地图中心坐标（max+min）/2

        for i in range(layers.shape[0]):
            layer_points[:, 2] = layers[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            if color is not None:
                layer_points[:, 3] = color[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            else:
                layer_points[:, 3] = 1.0
        
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, valid_points)
            pub_list[i].publish(points_msg) 

    def publishTomogram(self, layers_g, layers_t):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        n_slice = layers_g.shape[0]
        vis_g = layers_g.copy()
        vis_t = layers_t.copy() 
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        global_points = None
        for i in range(n_slice - 1): # 遍历层
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
            vis_g[i, mask_h] = np.nan  # 将当前层冗余区域（当前区域单元格与上层高度<ds）标记为NaN
            vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1, mask_h])  # 重叠区取最小代价
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]] # 高度索引 
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]] # 强度索引
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]  # 把layer_points非空的索引单元格的值赋给valid_points
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0) # 按行拼接点云数据

        # 上面for循环只处理到了倒数第二层，下面，处理最后一层
        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]] 
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]] 
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)
        
        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
    args = parser.parse_args()

    cfg = Config()
    scene_cfg = getattr(__import__('config'), 'Scene' + args.scene)

    rospy.init_node('pointcloud_tomography', anonymous=True)

    mapping = Tomography(cfg, scene_cfg)

    rospy.spin()