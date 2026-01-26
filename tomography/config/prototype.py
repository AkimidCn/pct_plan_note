import numpy as np
from sensor_msgs.msg import PointField


POINT_FIELDS_XYZI = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('intensity', 12, PointField.FLOAT32, 1)
]


def GRID_POINTS_XYZI(resolution, dim_x, dim_y):
    index_proto = np.zeros((dim_x * dim_y, 2), dtype=int)
    
    # 举例说明:
    # lx = [0, 1, 2]  
    # ly = [0, 1]
    lx = np.linspace(0, dim_x - 1, dim_x, dtype=int)   #创建一个从 0 到 dim_x-1 的等间隔整数序列，包含 dim_x 个元素。
    ly = np.linspace(0, dim_y - 1, dim_y, dtype=int)
    
    #ix = [ [0, 1, 2],   iy = [ [0, 0, 0],
    #       [0, 1, 2] ]         [1, 1, 1] ]
    ix, iy = np.meshgrid(lx, ly)
    
    # index_proto = [
    # [0, 0],  # 网格(0,0)
    # [1, 0],  # 网格(1,0)
    # [2, 0],  # 网格(2,0)
    # [0, 1],  # 网格(0,1) 
    # [1, 1],  # 网格(1,1)
    # [2, 1]   # 网格(2,1)
    # ]
    index_proto[:, 0] = ix.flatten()
    index_proto[:, 1] = iy.flatten()

    point_proto = np.zeros((dim_x * dim_y, 4), dtype=np.float32)
    point_proto[:, :2] = index_proto[:, :2].astype(np.float32, copy=True)
    
    point_proto[:, 0] -= 0.5 * dim_x  # 将坐标原点从左上角移动到中心
    point_proto[:, 1] -= 0.5 * dim_y
    
    point_proto[:, :2] *= resolution
    point_proto[:, 3] = 1.0

    return index_proto, point_proto