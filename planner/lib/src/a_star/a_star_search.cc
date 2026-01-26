#include "a_star/a_star_search.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using std::cout;
using std::endl;

// 9 neighbors in 2d
static std::vector<Eigen::Vector2i> kNeighbors = std::vector<Eigen::Vector2i>{
    Eigen::Vector2i(-1, -1), Eigen::Vector2i(-1, 0), Eigen::Vector2i(-1, 1),
    Eigen::Vector2i(0, -1),  Eigen::Vector2i(0, 1),  Eigen::Vector2i(1, -1),
    Eigen::Vector2i(1, 0),   Eigen::Vector2i(1, 1),
};

void Astar::Init(const double cost_threshold, const int num_layers,
                 const double resolution,  const double step_cost_weight, const Eigen::MatrixXd& cost_map,
                 const Eigen::MatrixXd& height_map,
                 const Eigen::MatrixXd& ele_map) {  // ele_map 网关地图
  auto t0 = std::chrono::high_resolution_clock::now();
  cost_threshold_ = cost_threshold;
step_cost_weight_  = step_cost_weight;

  max_x_ = cost_map.cols();     // 行
  max_y_ = cost_map.rows() / num_layers;
  max_layers_ = num_layers;
  xy_size_ = max_x_ * max_y_;

  int row_offset = 0;
  grid_map_.resize(max_layers_);  // 将grid_map_向量指定大小为max_layers_
  for (size_t i = 0; i < max_layers_; ++i) {
    row_offset = i * max_y_;
    grid_map_[i].resize(max_y_);
    for (size_t j = 0; j < max_y_; ++j) {
      grid_map_[i][j].resize(max_x_);
      for (size_t k = 0; k < max_x_; ++k) {
        double height = height_map(j + row_offset, k);
        double z = static_cast<int>(height / resolution);
        grid_map_[i][j][k] = Node(Eigen::Vector3i(z, j, k), nullptr);
        grid_map_[i][j][k].cost = cost_map(j + row_offset, k);
        grid_map_[i][j][k].height = height;
        grid_map_[i][j][k].ele = ele_map(j + row_offset, k);  // 网关
        grid_map_[i][j][k].layer = i;
      }
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - t0);

  search_layers_offset_.clear();   // 清空向量 → []
  search_layers_offset_.emplace_back(0);  // 添加0 → [0]
  for (int i = 0; i < search_layer_depth_; ++i) {
    search_layers_offset_.emplace_back(-(i + 1));
    search_layers_offset_.emplace_back(i + 1);
  } // search_layers_offset_  = [0, -1, 1]      定义搜索顺序，只能在相邻层进行搜索

  printf(
      "Astar initialized, max_x: %d, max_y: %d, max_layers: %d, time elapsed: "
      "%f ms\n",
      max_x_, max_y_, max_layers_, duration.count() / 1000.0);
}

void Astar::Reset() {
  for (size_t i = 0; i < grid_map_.size(); ++i) {
    for (size_t j = 0; j < grid_map_[i].size(); ++j) {
      for (size_t k = 0; k < grid_map_[i][j].size(); ++k) {
        grid_map_[i][j][k].Reset();
      }
    }
  }
}

int Astar::GetHash(const Eigen::Vector3i& idx) const {
  return idx[0] * 10000000 + idx[1] * max_x_ + idx[2];
}

bool Astar::Search(const Eigen::Vector3i& start, const Eigen::Vector3i& goal) {
  auto t0 = std::chrono::high_resolution_clock::now();

  if (!search_result_.empty()) {
    Reset();
    search_result_.clear();
  }

  auto start_node = &grid_map_[start[0]][start[2]][start[1]];  // start[0]=layer; start[1]=x; start[2]=y
  auto goal_node = &grid_map_[goal[0]][goal[2]][goal[1]];
  start_node->g = 0.0;

  if (goal_node->cost > cost_threshold_) {
    printf("goal node is not reachable, cost: %f", goal_node->cost);
    return false;
  }
  //优先队列容器  存储的元素类型：Node指针；// 底层容器：使用vector存储Node指针；  // 比较器：自定义的排序规则
  std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_set;  // 待确定点
  std::unordered_map<int, Node*> closed_set; // 已确定点

  open_set.push(start_node);

  printf("start searching\n");

  while (!open_set.empty()) {
    Node* current_node = open_set.top();
    open_set.pop();  // 

    if (current_node->idx == goal_node->idx) {  // 如果是目标点
      while (current_node->parent != nullptr) {  //回溯找父节点
        // search_result_.emplace_back(Eigen::Vector3i(
        //     current_node->layer, current_node->idx[1],
        //     current_node->idx[2]));
        search_result_.emplace_back(current_node);
        current_node = current_node->parent;
      }
      std::reverse(search_result_.begin(), search_result_.end()); // 回溯得到的是逆序路径，需要反转
      if (debug_) ConvertClosedSetToMatrix(closed_set);
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t0);
      printf("path found, time elapsed: %f ms\n",
             duration.count() / 1000.0);
      return true;
    }

    closed_set[GetHash(current_node->idx)] = current_node;   // 将当前节点加入closed_set列表； unordered_map的存储方式只会分配对应键值（closed_set[key]）的内存空间

    // int layer = current_node->layer;
    // if (current_node->ele > 0.5) {
    //   layer = std::min(layer + 1, max_layers_ - 1);
    // } else if (current_node->ele < -0.5) {
    //   layer = std::max(layer - 1, 0);
    // }
    int layer = DecideLayer(current_node); //网关检测和层切换决策

    int i, j = 0;
    double tentative_g = 0.0;
    for (const auto& neighbor : kNeighbors) { // 遍历同一层的8个邻居节点
      i = current_node->idx[1] + neighbor[0];
      j = current_node->idx[2] + neighbor[1];

      if (i < 0 || i >= max_y_ || j < 0 || j >= max_x_) {
        continue;
      }

      auto neighbor_node = &grid_map_[layer][i][j];

      if (neighbor_node->cost > cost_threshold_) {
        if (abs(neighbor_node->ele) < 0.5) {
          continue;
        } else {
          if (std::abs(neighbor_node->height - current_node->height) > 0.3) {
            continue;
          }
        }
      }

      // if ((neighbor_node->cost > cost_threshold_) ||
      //     std::abs(neighbor_node->height - current_node->height) > 0.3) {
      //   continue;
      // }

      auto diff = neighbor_node->idx - current_node->idx;  //计算当前节点与邻居节点在三维空间中的网格差
      double step_cost = step_cost_weight_ * neighbor_node->cost; //当前代码设置step_cost_weight_ = 1
      if (step_cost < 5) step_cost = 0.0;
      tentative_g =  //到邻居节点的估计代价
          current_node->g +  // 当前累计代价
          std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) +  // 当前节点到邻居节点的三维欧几里得距离
          step_cost; //基于地图通行难度的额外代价

      auto p_neighbor = closed_set.find(GetHash(neighbor_node->idx));
      if (p_neighbor != closed_set.end()) {  //closed_set已有；end() 指向的是容器的最后一个位置的下一个位置（类似指向结束符）
        if (tentative_g >= p_neighbor->second->g) {  // astar算法中比较g的大小
          continue;
        }
      }

      if (tentative_g < neighbor_node->g) {  // 当前节点到邻居节点代价<邻居当前代价；更新邻居节点的g、f值和父节点指针
        neighbor_node->g = tentative_g;
        neighbor_node->f = tentative_g + GetHeuristic(neighbor_node, goal_node);
        neighbor_node->parent = current_node;
        open_set.push(neighbor_node);
      }
    }
  }

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - t0);
  printf("path not found\n, time elapsed: %f ms\n",
         duration.count() / 1000.0);
  if (debug_) {
    ConvertClosedSetToMatrix(closed_set);
  }
  return false;
}

int Astar::DecideLayer(const Node* cur_node) const {
  int layer = cur_node->layer;
  int i = cur_node->idx[1];  //row
  int j = cur_node->idx[2];  //col
  double cur_height = cur_node->height;

  int true_layer = layer;

  for (const auto offset : search_layers_offset_) {  // 遍历预设的层偏移序列 [0, -1, 1]
    int cur_layer = layer + offset;

    if (cur_layer < 0 || cur_layer >= max_layers_) {
      continue;
    }

    const Node& search_node = grid_map_[cur_layer][i][j];

    if (abs(search_node.height - cur_height) > 0.2) {
      continue;
    }

    if (search_node.ele > 0.5) {
      true_layer = std::min(cur_layer + 1, max_layers_ - 1);  //向上层搜索
      break;
    } else if (search_node.ele < -0.5) {
      true_layer = std::max(cur_layer - 1, 0);  // 向下层搜索
      break;
    }
  }

  return true_layer;
}

double Astar::CalculateStepCost(const Node* node1, const Node* node2) const {}

double Astar::GetHeuristic(const Node* node1, const Node* node2) const {
  double cost = 0.0;

  if (h_type_ == kEuclidean) {
    // l2 distance
    cost = (node1->idx - node2->idx).norm();
  } else if (h_type_ == kDiagonal) {
    // octile distance
    Eigen::Vector3i d = node1->idx - node2->idx;
    int dx = abs(d(0)), dy = abs(d(1)), dz = abs(d(2));
    int dmin = std::min(dx, std::min(dy, dz));
    int dmax = std::max(dx, std::max(dy, dz));
    int dmid = dx + dy + dz - dmin - dmax;
    double h =
        std::sqrt(3) * dmin + std::sqrt(2) * (dmid - dmin) + (dmax - dmid);
    cost = h;
  } else if (h_type_ == kManhattan) {
    cost = (node1->idx - node2->idx).lpNorm<1>();
  } else {
    assert(false && "not implemented");
  }

  // cost += std::abs(node1->idx[0] - node2->idx[0]) * 10;
  return cost;
}

std::vector<PathPoint> Astar::GetPathPoints() const {
  std::vector<PathPoint> path_points;

  auto size = search_result_.size();
  path_points.resize(size);

  if (size == 0) {
    printf("path is empty\n, convert to path points failed\n");
    return path_points;
  }

  for (size_t i = 0; i < size; ++i) {
    // path_points[i].layer = search_result_[i][0];
    // path_points[i].x = search_result_[i][2];
    // path_points[i].y = search_result_[i][1];
    // if (i > 0) {
    //   path_points[i].heading =
    //       std::atan2(search_result_[i][1] - search_result_[i - 1][1],
    //                  search_result_[i][2] - search_result_[i - 1][2]);
    // }
    path_points[i].layer = search_result_[i]->layer;
    path_points[i].x = search_result_[i]->idx(2);
    path_points[i].y = search_result_[i]->idx(1);
    path_points[i].height = search_result_[i]->height;
    if (i > 0) {
      path_points[i].heading =
          std::atan2(search_result_[i]->idx(1) - search_result_[i - 1]->idx(1),
                     search_result_[i]->idx(2) - search_result_[i - 1]->idx(2));
    }
  }

  if (size > 1) {
    path_points[0].heading = path_points[1].heading;
  }

  return path_points;
}

Eigen::MatrixXd Astar::GetResultMatrix() const {
  if (search_result_.empty()) {
    printf("path is empty\n, convert to matrix failed\n");
    return Eigen::MatrixXd();
  }

  Eigen::MatrixXd path_matrix(search_result_.size(), 3);
  for (size_t i = 0; i < search_result_.size(); ++i) {
    path_matrix(i, 0) = search_result_[i]->layer;
    path_matrix(i, 1) = search_result_[i]->idx[1];
    path_matrix(i, 2) = search_result_[i]->idx[2];
  }
  return path_matrix;
}

void Astar::ConvertClosedSetToMatrix(
    const std::unordered_map<int, Node*>& closed_set) {
  visited_set_ = Eigen::MatrixXi(closed_set.size(), 3);
  int count = 0;
  for (auto i = closed_set.begin(); i != closed_set.end(); ++i) {
    visited_set_(count, 0) = i->second->layer;
    visited_set_(count, 1) = i->second->idx[1];
    visited_set_(count, 2) = i->second->idx[2];
    count += 1;
  }
}

std::vector<Eigen::Vector3i> Astar::GetNeighbors(Node* node) const {}

Eigen::MatrixXd Astar::GetCostLayer(int layer) const {
  Eigen::MatrixXd cost_layer(max_y_, max_x_);
  for (int i = 0; i < max_y_; ++i) {
    for (int j = 0; j < max_x_; ++j) {
      cost_layer(i, j) = grid_map_[layer][i][j].cost;
    }
  }
  return cost_layer;
}
Eigen::MatrixXd Astar::GetEleLayer(int layer) const {
  Eigen::MatrixXd ele_layer(max_y_, max_x_);
  for (int i = 0; i < max_y_; ++i) {
    for (int j = 0; j < max_x_; ++j) {
      ele_layer(i, j) = grid_map_[layer][i][j].ele;
    }
  }
  return ele_layer;
}