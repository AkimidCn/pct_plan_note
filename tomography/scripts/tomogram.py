import numpy as np
import cupy as cp

from kernels import *


class Tomogram(object):
    def __init__(self, cfg):
        self.resolution = cfg.map.resolution
        self.slice_dh = cfg.map.slice_dh
        self.half_trav_k_size = int(cfg.trav.kernel_size / 2)
        self.interval_min = cfg.trav.interval_min
        self.interval_free = cfg.trav.interval_free
        self.step_stand = 1.2 * self.resolution * np.tan(cfg.trav.slope_max)
        self.step_cross = cfg.trav.step_max
        self.standable_th = int(cfg.trav.standable_ratio * (2 * self.half_trav_k_size + 1) ** 2) - 1
        self.cost_barrier = float(cfg.trav.cost_barrier)
        self.safe_margin = cfg.trav.safe_margin
        self.inflation = cfg.trav.inflation
        self.half_inf_k_size = int((self.safe_margin + self.inflation) / self.resolution)    # 膨胀核 = 安全距离+膨胀距离

    def initKernel(self):
        self.tomography_kernel = tomographyKernel(
            self.resolution, 
            self.map_dim_x, 
            self.map_dim_y,
            self.n_slice_init,
            self.slice_h0,
            self.slice_dh
        )

        self.trav_kernel = travKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_trav_k_size,
            self.interval_min,
            self.interval_free,
            self.step_cross, 
            self.step_stand, 
            self.standable_th, 
            self.cost_barrier
        )

        self.inflation_kernel = inflationKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_inf_k_size
        )

        # 膨胀核的权重表，公式6的  min部分
        self.inf_table = cp.zeros(
            (2 * self.half_inf_k_size + 1, 2 * self.half_inf_k_size + 1), 
            dtype=cp.float32
        )
        for i in range(self.inf_table.shape[0]):
            for j in range(self.inf_table.shape[1]):
                # dist为d_mn   核上点到中心点的距离
                dist = np.sqrt(
                    (self.resolution * (i - self.half_inf_k_size)) ** 2 + \
                    (self.resolution * (j - self.half_inf_k_size)) ** 2
                )    
                self.inf_table[i, j] = np.clip(
                    1 - (dist - self.inflation) / (self.safe_margin + self.resolution),
                    a_min=0.0, a_max=1.0 
                )

    def initBuffers(self):
        self.layers_g = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.layers_c = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.grad_mag_sq = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.grad_mag_max = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.trav_cost = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.inflated_cost = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)

    def initMappingEnv(self, center, map_dim_x, map_dim_y, n_slice_init, slice_h0):
        self.center = cp.array(center, dtype=cp.float32)
        self.map_dim_x = int(map_dim_x)
        self.map_dim_y = int(map_dim_y)
        self.n_slice_init = int(n_slice_init)
        self.slice_h0 = float(slice_h0)
        
        self.initBuffers()
        self.initKernel()   # 这里是拿到kernel函数

    def clearMap(self):
        self.layers_g *= 0.
        self.layers_c *= 0.
        self.layers_g += -1e6
        self.layers_c += 1e6

        self.grad_mag_sq *= 0.
        self.grad_mag_max *= 0.
        self.trav_cost *= 0.
        self.inflated_cost *= 0.

    def point2map(self, points):
        points = cp.asarray(points)   # 转换为CuPy数组（GPU加速）
        points = points[~cp.isnan(points).any(axis=1)]  # 移除包含NaN值的点
        self.clearMap()

        # Tomogram
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()
        
        ## 获取各切片的 g，c,对应论文算法1的，5、6行
        self.tomography_kernel(
            points, self.center, 
            self.layers_g, self.layers_c,
            size=(points.shape[0])
        )
        
        # 梯度计算
        # layers_g[切片层数, x, y]
        diff_x_sq = cp.maximum(
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2, 
            (self.layers_g[:, 1:-1, :] - self.layers_g[:,  2:, :]) ** 2
        )
        diff_y_sq = cp.maximum(
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2, 
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :,  2:]) ** 2
        )
        # 论文公式(2) m^grad
        self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
        # 找二者最大值给grad_mag_max：m^xy = max(|g^x|,|g^y|)，这里是平方值
        self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :])
        
        # 天花板于地面的高度差 d^I = e^C - e^G
        interval = (self.layers_c - self.layers_g)

        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_map = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        # Traversability
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()

        # 论文公式 2-4 代价计算
        self.trav_kernel(
            interval, self.grad_mag_sq, self.grad_mag_max,
            self.trav_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        # 膨胀核与代价地图的hadamard积——矩阵元素逐点相乘
        # inflated_cost输出为c^init
        self.inflation_kernel(
            self.trav_cost, self.inf_table,
            self.inflated_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_trav = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        # Layer Simplification   层 简化
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()

        idx_simp = [0]
        if self.layers_g.shape[0] > 1:   # 切片数大于1
            l_idx, m_idx = 0, 1   # m_idx是当前考察的切片，l_idx是上一个被选中的切片
            diff_h = self.layers_g[1:] - self.layers_g[:-1]   # 计算 相邻 切片高度差
            while m_idx < self.n_slice_init - 2:    # 第二个 到 倒数第三个  切片
                
                # 公式8  空间位置唯一性判断 e^G_k - e^G_k-1 > 0
                mask_l_g = self.layers_g[m_idx] - self.layers_g[l_idx] > 0
                
                # 公式8 代价 唯一性判断
                mask_l_t = self.inflated_cost[l_idx] > self.inflated_cost[m_idx]
                
                # 相邻的层
                mask_u_g = diff_h[m_idx] > 0
                
                # 可穿越网格判断  <c^B
                mask_t = self.inflated_cost[m_idx] < self.cost_barrier
                
                unique = (mask_l_g | mask_l_t) & mask_u_g & mask_t
                if cp.any(unique):    # 检查整个数组中是否存在至少一个 True 值
                    idx_simp.append(m_idx) # 将m_idx加入idx_simp列表
                    l_idx = m_idx
                m_idx += 1
            idx_simp.append(m_idx)

        # 先选中idx_simp中的切片，再选每个切片的行或列；inflated_cost[idx_simp][:, 2:, :]应该就等于inflated_cost[idx_simp, 2:, :]
        trav_grad_x = (self.inflated_cost[idx_simp][:, 2:, :] - self.inflated_cost[idx_simp][:, :-2, :])
        trav_grad_y = (self.inflated_cost[idx_simp][:, :, 2:] - self.inflated_cost[idx_simp][:, :, :-2])
        
        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_simp = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        
        gpu_t_all = gpu_t_map + gpu_t_trav + gpu_t_simp
        #print("CuPy GPU time (ms):", gpu_t_all)

        layers_t = self.inflated_cost[idx_simp].get()  # 这里是取出再GPU内存中的数据
        
        # cp.where(条件, 真时取值, 假时取值)
        layers_g = cp.where(
            self.layers_g[idx_simp] > -1e6, 
            self.layers_g[idx_simp],
            cp.nan
        ).get()
        layers_c = cp.where(
            self.layers_c[idx_simp] < 1e6, 
            self.layers_c[idx_simp], 
            cp.nan
        ).get()
        trav_gx = np.zeros_like(layers_g)  # 创建一个与layers_g形状和数据类型相同的新数组
        trav_gx[:, 1:-1, :] = trav_grad_x.get()
        trav_gy = np.zeros_like(layers_g)
        trav_gy[:, :, 1:-1] = trav_grad_y.get()

        t_gpu = {
            't_map': gpu_t_map, 
            't_trav': gpu_t_trav, 
            't_simp': gpu_t_simp, 
        }
        
        return layers_t, trav_gx, trav_gy, layers_g, layers_c, t_gpu