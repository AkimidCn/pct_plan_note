import string
import cupy as cp


def utils_point(resolution, n_row, n_col):
    util_preamble = string.Template(
        '''
        __device__ int getIndexLine(float16 x, float16 center)
        {
            int i = round((x - center) / ${resolution});
            return i;
        }

        __device__ int getIndexMap_1d(float16 x, float16 y, float16 cx, float16 cy)
        {
            // Return 1D index of a point (x, y) in a layer
            int idx_x = getIndexLine(x, cx) + ${n_row} / 2;         
            int idx_y = getIndexLine(y, cy) + ${n_col} / 2;

            // 是否在地图中
            if (idx_x < 0 || idx_x >= ${n_row} || idx_y < 0 || idx_y >= ${n_col})
            {
                return -1;
            }
            return ${n_col} * idx_x + idx_y;
        }

        __device__ int getIndexBlock_1d(int idx, int layer_n)
        {
            // Return 1D index of a point (x, y) in multi-layer map block
            return (int)${layer_size} * layer_n + idx;
        }

        __device__ static float atomicMaxFloat(float* address, float val)
        {
            int* address_as_i = (int*) address;
            int old = *address_as_i, assumed;  //old是指针指向的值
            do {
                assumed = old;
                
                //这一步是将address_as_i地址的值与assumed对比，若相同，则将__float_as_int的值写入address_as_i地址中***
                //主要是避免多线程冲突
                old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fmaxf(val, __int_as_float(assumed))));
            } while (assumed != old);

            return __int_as_float(old);
        }

        __device__ static float atomicMinFloat(float* address, float val)
        {
            int* address_as_i = (int*) address;
            int old = *address_as_i, assumed;
            do {
                assumed = old;
                old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fminf(val, __int_as_float(assumed))));
            } while (assumed != old);

            return __int_as_float(old);
        }
        '''
    ).substitute(
        resolution=resolution,
        n_row=n_row, 
        n_col=n_col,
        layer_size=n_row*n_col
    )

    return util_preamble


def utils_map(n_row, n_col):
    util_preamble=string.Template(
        '''
        __device__ int getIdxRelative(int idx, int dx, int dy) 
        {
            // Return 1D index of the relative point (x+dx, y+dy) in multi-layer map block
            int idx_2d = idx % (int)${layer_size};
            int idx_x = idx_2d / ${n_col};
            int idx_y = idx_2d % ${n_col};
            int idx_rx = idx_x + dx;
            int idx_ry = idx_y + dy;

            if ( idx_rx < 0 || idx_rx > (${n_row} - 1) ) 
                return -1;
            if ( idx_ry < 0 || idx_ry > (${n_col} - 1) )
                return -1;

            return ${n_col} * dx + dy + idx;
        }
        '''
    ).substitute(
        n_row=n_row, 
        n_col=n_col,
        layer_size=n_row*n_col
    )

    return util_preamble


def tomographyKernel(resolution, n_row, n_col, n_slice, slice_h0, slice_dh): 
    tomography_kernel = cp.ElementwiseKernel(           # CuPy提供的并行流水线模板   "创建一个流水线，让每个工人独立处理一个数据元素"
        in_params='raw U points, raw U center',         # 输入参数  raw是传递数组的内存地址
        out_params='raw U layers_g, raw U layers_c',    # 输出参数
        preamble=utils_point(resolution, n_row, n_col),
        
        # atomicMinFloat(&layers_c[getIndexBlock_1d(idx, s_idx)], pz);
        operation=string.Template(
            '''
            U px = points[i * 3];          // Cupy将其视为一段连续的一维内存地址
            U py = points[i * 3 + 1];
            U pz = points[i * 3 + 2];

            //以一维方式输出当前点在地图(size=map_dim_x*map_dim_y)的索引:  行索引*行数+列索引
            int idx = getIndexMap_1d(px, py, center[0], center[1]);  
            if ( idx < 0 ) 
                return; 
            for ( int s_idx = 0; s_idx < ${n_slice}; s_idx ++ )
            {
                U slice = ${slice_h0} + s_idx * ${slice_dh};
                if ( pz <= slice )
                   //getIndexBlock_1d是返回加上层slice的索引：层网格数 * 当前层数 + idx
                    atomicMaxFloat(&layers_g[getIndexBlock_1d(idx, s_idx)], pz);
                else
                    atomicMinFloat(&layers_c[getIndexBlock_1d(idx, s_idx)], pz);
            }
            '''
        ).substitute(
            n_slice=n_slice,
            slice_h0=slice_h0,
            slice_dh=slice_dh
        ),
        name='tomography_kernel'
    )
                            
    return tomography_kernel


def travKernel(
    n_row, n_col, half_kernel_size,
    interval_min, interval_free, step_cross, step_stand, standable_th, cost_barrier
    ):
    trav_kernel = cp.ElementwiseKernel(
        in_params='raw U interval, raw U grad_mag_sq, raw U grad_mag_max',
        out_params='raw U trav_cost',
        preamble=utils_map(n_row, n_col),
        operation=string.Template(
            '''
            // 论文公式1 代价 c^I
            if ( interval[i] < ${interval_min} )
            {
                trav_cost[i] = ${cost_barrier};  // 不可通行
                return;
            }
            else
                trav_cost[i] += max(0.0, 20 * (${interval_free} - interval[i]));
                
            // 论文公式3 代价 c^G 条件2 
            if ( grad_mag_sq[i] <= ${step_stand_sq} )
            {
                trav_cost[i] += 15 * grad_mag_sq[i] / ${step_stand_sq};
                return;
            }
            else 
            {
                if ( grad_mag_max[i] <= ${step_cross_sq} )
                {
                    int standable_grids = 0;
                    // half_kernel_size  为邻域大小的一半 ,邻域原为7
                    for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
                    {
                        for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
                        {
                            int idx = getIdxRelative(i, dx, dy);
                            if ( idx < 0 )
                                continue;
                            if ( grad_mag_sq[idx] < ${step_stand_sq} )
                                standable_grids += 1;
                        }
                    }
                    // 论文公式4 条件2
                    if ( standable_grids < ${standable_th} )
                    {
                        trav_cost[i] = ${cost_barrier};
                        return;
                    }
                    else // 论文公式4 条件1
                        trav_cost[i] += 20 * grad_mag_max[i] / ${step_cross_sq};
                }
                else
                {
                    trav_cost[i] = ${cost_barrier};
                    return;
                }
            } 
            '''
        ).substitute(
            half_kernel_size=half_kernel_size,
            interval_min=interval_min,
            interval_free=interval_free,
            step_cross_sq=step_cross ** 2,
            step_stand_sq=step_stand ** 2,
            standable_th=standable_th,
            cost_barrier=cost_barrier
        ),
        name='trav_kernel'
    )
                            
    return trav_kernel


def inflationKernel(n_row, n_col, half_kernel_size):
    inflation_kernel = cp.ElementwiseKernel(
        in_params='raw U trav_cost, raw U score_table',
        out_params='raw U inflated_cost',
        preamble=utils_map(n_row, n_col),
        operation=string.Template(
            '''
            int counter = 0;
            for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
            {
                for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
                {
                    int idx = getIdxRelative(i, dx, dy);  // 这里的idx是相对于当前点i的邻域点索引，即(i+dx, j+dy)在地图中的1维索引
                    if ( idx >= 0 )
                        inflated_cost[i] = max(inflated_cost[i], trav_cost[idx] * score_table[counter]);  //这里trav_cost 就是论文中的c^init
                    counter += 1;
                }
            }
            '''
        ).substitute(
            half_kernel_size=half_kernel_size
        ),
        name='inflation_kernel'
    )
                            
    return inflation_kernel