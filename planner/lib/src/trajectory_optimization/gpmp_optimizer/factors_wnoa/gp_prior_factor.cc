#include "trajectory_optimization/gpmp_optimizer/factors_wnoa/gp_prior_factor.h"

using gtsam::Matrix;

gtsam::Vector GPPriorFactorWnoa::evaluateError(
    const gtsam::Vector4& x1, const gtsam::Vector4& x2,
    boost::optional<gtsam::Matrix&> H1,
    boost::optional<gtsam::Matrix&> H2) const {
  if (H1) *H1 = phi_;
  if (H2) *H2 = -gtsam::Matrix44::Identity();
  return phi_ * x1 - x2;   // phi_ * x1是通过运动学预测的位置   x2是优化器给出的实际状态
}
