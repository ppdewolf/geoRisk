// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppParallel)]]

#include <Rcpp.h>
#include <RcppParallel.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

struct Point {
  double x;
  double y;
  int w;
};

static inline double sqr(double x) { return x * x; }

static inline double dist2_xy(double ax, double ay, double bx, double by) {
  return sqr(ax - bx) + sqr(ay - by);
}

static inline bool leq_dist2(double d2, double r2, double eps) {
  return d2 <= r2 + eps * (1.0 + r2);
}

struct KDNode {
  int idx;
  int left;
  int right;
  double minx;
  double maxx;
  double miny;
  double maxy;
  long long weight_sum;
};

// KD-tree over unique weighted locations. It is used for exact nearest-neighbor
// upper radii and for summing all weights inside the final centered circle.
class KDTree {
public:
  explicit KDTree(const std::vector<Point>& points) : pts_(&points), root_(-1) {
    ids_.resize(points.size());
    std::iota(ids_.begin(), ids_.end(), 0);
    nodes_.reserve(points.size());
    root_ = build(0, static_cast<int>(ids_.size()), 0);
  }

  std::vector<int> knearest(double x, double y, int k, double eps) const {
    std::priority_queue<std::pair<double, int> > heap;
    knearest_rec(root_, x, y, k, eps, heap);

    std::vector<std::pair<double, int> > tmp;
    tmp.reserve(heap.size());
    while (!heap.empty()) {
      tmp.push_back(heap.top());
      heap.pop();
    }
    std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });

    std::vector<int> ans;
    ans.reserve(tmp.size());
    for (const auto& item : tmp) ans.push_back(item.second);
    return ans;
  }

  long long range_weight(double x, double y, double r, double eps) const {
    return range_weight_rec(root_, x, y, r * r, eps);
  }

private:
  const std::vector<Point>* pts_;
  std::vector<int> ids_;
  std::vector<KDNode> nodes_;
  int root_;

  int build(int lo, int hi, int depth) {
    if (lo >= hi) return -1;
    const int mid = lo + (hi - lo) / 2;
    const int axis = depth & 1;
    auto comp = [&](int a, int b) {
      const Point& pa = (*pts_)[a];
      const Point& pb = (*pts_)[b];
      if (axis == 0) {
        if (pa.x != pb.x) return pa.x < pb.x;
        return pa.y < pb.y;
      }
      if (pa.y != pb.y) return pa.y < pb.y;
      return pa.x < pb.x;
    };
    std::nth_element(ids_.begin() + lo, ids_.begin() + mid, ids_.begin() + hi, comp);

    const int node_id = static_cast<int>(nodes_.size());
    const int pidx = ids_[mid];
    nodes_.push_back(KDNode{pidx, -1, -1,
                            (*pts_)[pidx].x, (*pts_)[pidx].x,
                            (*pts_)[pidx].y, (*pts_)[pidx].y,
                            (*pts_)[pidx].w});

    const int left = build(lo, mid, depth + 1);
    const int right = build(mid + 1, hi, depth + 1);
    nodes_[node_id].left = left;
    nodes_[node_id].right = right;

    pull_child(node_id, left);
    pull_child(node_id, right);
    return node_id;
  }

  void pull_child(int node_id, int child_id) {
    if (child_id < 0) return;
    KDNode& n = nodes_[node_id];
    const KDNode& c = nodes_[child_id];
    n.minx = std::min(n.minx, c.minx);
    n.maxx = std::max(n.maxx, c.maxx);
    n.miny = std::min(n.miny, c.miny);
    n.maxy = std::max(n.maxy, c.maxy);
    n.weight_sum += c.weight_sum;
  }

  static double bbox_min_dist2(const KDNode& n, double x, double y) {
    double dx = 0.0;
    if (x < n.minx) dx = n.minx - x;
    else if (x > n.maxx) dx = x - n.maxx;

    double dy = 0.0;
    if (y < n.miny) dy = n.miny - y;
    else if (y > n.maxy) dy = y - n.maxy;

    return dx * dx + dy * dy;
  }

  static double bbox_max_dist2(const KDNode& n, double x, double y) {
    const double dx = std::max(std::fabs(x - n.minx), std::fabs(x - n.maxx));
    const double dy = std::max(std::fabs(y - n.miny), std::fabs(y - n.maxy));
    return dx * dx + dy * dy;
  }

  long long range_weight_rec(int node_id, double x, double y, double r2, double eps) const {
    if (node_id < 0) return 0;
    const KDNode& n = nodes_[node_id];
    if (!leq_dist2(bbox_min_dist2(n, x, y), r2, eps)) return 0;
    if (leq_dist2(bbox_max_dist2(n, x, y), r2, eps)) return n.weight_sum;

    long long ans = 0;
    const Point& p = (*pts_)[n.idx];
    if (leq_dist2(dist2_xy(x, y, p.x, p.y), r2, eps)) ans += p.w;
    ans += range_weight_rec(n.left, x, y, r2, eps);
    ans += range_weight_rec(n.right, x, y, r2, eps);
    return ans;
  }

  void offer_neighbor(double d2, int idx, int k,
                      std::priority_queue<std::pair<double, int> >& heap) const {
    if (static_cast<int>(heap.size()) < k) {
      heap.push(std::make_pair(d2, idx));
    } else if (d2 < heap.top().first ||
               (d2 == heap.top().first && idx < heap.top().second)) {
      heap.pop();
      heap.push(std::make_pair(d2, idx));
    }
  }

  void knearest_rec(int node_id, double x, double y, int k, double eps,
                    std::priority_queue<std::pair<double, int> >& heap) const {
    if (node_id < 0 || k <= 0) return;
    const KDNode& n = nodes_[node_id];

    const double lb = bbox_min_dist2(n, x, y);
    if (static_cast<int>(heap.size()) >= k &&
        lb > heap.top().first + eps * (1.0 + heap.top().first)) {
      return;
    }

    const Point& p = (*pts_)[n.idx];
    offer_neighbor(dist2_xy(x, y, p.x, p.y), n.idx, k, heap);

    const int left = n.left;
    const int right = n.right;
    const double dl = (left < 0) ? std::numeric_limits<double>::infinity()
                                 : bbox_min_dist2(nodes_[left], x, y);
    const double dr = (right < 0) ? std::numeric_limits<double>::infinity()
                                  : bbox_min_dist2(nodes_[right], x, y);

    if (dl < dr) {
      knearest_rec(left, x, y, k, eps, heap);
      knearest_rec(right, x, y, k, eps, heap);
    } else {
      knearest_rec(right, x, y, k, eps, heap);
      knearest_rec(left, x, y, k, eps, heap);
    }
  }
};

struct SolveResult {
  double radius;
  double weight;
};

static SolveResult solve_centered_one(int anchor,
                                      const std::vector<Point>& pts,
                                      const KDTree& tree,
                                      int k,
                                      double eps) {
  const Point& center = pts[anchor];

  // Since the center is fixed, the optimal radius is simply the smallest
  // distance threshold around the center whose enclosed total weight reaches k.
  const int nn_count = std::min(static_cast<int>(pts.size()), k);
  std::vector<int> nn = tree.knearest(center.x, center.y, nn_count, eps);

  long long cumulative = 0;
  double r2 = 0.0;
  for (int id : nn) {
    cumulative += pts[id].w;
    r2 = dist2_xy(center.x, center.y, pts[id].x, pts[id].y);
    if (cumulative >= k) break;
  }

  const double radius = std::sqrt(r2);

  // Count all points on or inside this exact radius, including ties at the
  // boundary. Therefore the returned weight may be greater than k.
  const long long total_weight = tree.range_weight(center.x, center.y, radius, eps);
  return SolveResult{radius, static_cast<double>(total_weight)};
}

class CenteredWorker : public RcppParallel::Worker {
public:
  CenteredWorker(const std::vector<Point>& pts,
                 const KDTree& tree,
                 int k,
                 double eps,
                 NumericVector radius,
                 NumericVector out_weight)
      : pts_(pts), tree_(tree), k_(k), eps_(eps),
        radius_(radius), out_weight_(out_weight) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      SolveResult ans = solve_centered_one(static_cast<int>(i), pts_, tree_, k_, eps_);
      radius_[i] = ans.radius;
      out_weight_[i] = ans.weight;
    }
  }

private:
  const std::vector<Point>& pts_;
  const KDTree& tree_;
  int k_;
  double eps_;
  RcppParallel::RVector<double> radius_;
  RcppParallel::RVector<double> out_weight_;
};

// U is the original multiset: every row is one occurrence of a location.
// Equal coordinate pairs are aggregated internally into unique weighted
// locations. The output is expanded back to one row per original input row.
//
// For each target location p, this function returns the smallest circle whose
// center is fixed at p and whose enclosed total weight is at least k.
//
// [[Rcpp::export]]
Rcpp::DataFrame centered_weighted_circles(const NumericMatrix& U,
                                          int k,
                                          bool parallel = true,
                                          double eps = 1e-10) {
  if (U.ncol() != 2) Rcpp::stop("U must have exactly two columns.");
  if (k < 1) Rcpp::stop("k must be >= 1.");

  const int n = U.nrow();
  if (n == 0) Rcpp::stop("U must contain at least one row.");
  if (n < k) Rcpp::stop("nrow(U) must be >= k.");

  std::vector<Point> pts;
  pts.reserve(n);
  std::vector<int> original_to_unique(n);
  std::map<std::pair<double, double>, int> id_by_coord;

  for (int i = 0; i < n; ++i) {
    const double x = U(i, 0);
    const double y = U(i, 1);
    if (!R_finite(x) || !R_finite(y)) {
      Rcpp::stop("U contains non-finite coordinates.");
    }

    const std::pair<double, double> key(x, y);
    auto it = id_by_coord.find(key);
    if (it == id_by_coord.end()) {
      const int id = static_cast<int>(pts.size());
      id_by_coord[key] = id;
      original_to_unique[i] = id;
      pts.push_back(Point{x, y, 1});
    } else {
      original_to_unique[i] = it->second;
      pts[it->second].w += 1;
    }
  }

  KDTree tree(pts);

  const int m = static_cast<int>(pts.size());
  NumericVector unique_radius(m), unique_weight(m);
  CenteredWorker worker(pts, tree, k, eps, unique_radius, unique_weight);

  if (parallel) {
    RcppParallel::parallelFor(0, m, worker);
  } else {
    worker(0, m);
  }

  NumericVector px(n), py(n), cx(n), cy(n), radius(n), out_weight(n);
  for (int i = 0; i < n; ++i) {
    const int id = original_to_unique[i];
    px[i] = U(i, 0);
    py[i] = U(i, 1);
    cx[i] = U(i, 0);
    cy[i] = U(i, 1);
    radius[i] = unique_radius[id];
    out_weight[i] = unique_weight[id];
  }

  return Rcpp::DataFrame::create(
    Rcpp::Named("px") = px,
    Rcpp::Named("py") = py,
    Rcpp::Named("cx") = cx,
    Rcpp::Named("cy") = cy,
    Rcpp::Named("radius") = radius,
    Rcpp::Named("weight") = out_weight
  );
}

