// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppParallel)]]

#include <Rcpp.h>
#include <RcppParallel.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

using Rcpp::IntegerVector;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

// This file solves the following anchored problem.
// For each requested anchor p, find the smallest circle that contains p and
// contains total input weight at least k. The anchor does not have to be the
// center and does not have to lie on the boundary.
//
// The input multiset is compressed to unique coordinates. The integer weight
// of a unique coordinate is its multiplicity in the original multiset.
struct Point {
  double x;
  double y;
  int w;
};

// A candidate circle. The support field records how many points generated it
// (1, 2, or 3); it is diagnostic and not needed for correctness.
struct Circle {
  double x;
  double y;
  double r;
  int support;
  bool ok;
};

static inline double sqr(double x) { return x * x; }

static inline double dist2(const Point& a, const Point& b) {
  return sqr(a.x - b.x) + sqr(a.y - b.y);
}

static inline double dist2_xy(double ax, double ay, double bx, double by) {
  return sqr(ax - bx) + sqr(ay - by);
}

static inline bool leq_dist2(double d2, double r2, double eps) {
  return d2 <= r2 + eps * (1.0 + r2);
}

static inline bool point_in_circle(const Point& p, const Circle& c, double eps) {
  if (!c.ok) return false;
  return leq_dist2(dist2_xy(p.x, p.y, c.x, c.y), c.r * c.r, eps);
}

// The only circles that can be minimal enclosing circles in the plane are
// determined by one point, two antipodal points, or three boundary points.
static Circle circle_one(const Point& a) {
  return Circle{a.x, a.y, 0.0, 1, true};
}

static Circle circle_two(const Point& a, const Point& b) {
  const double cx = 0.5 * (a.x + b.x);
  const double cy = 0.5 * (a.y + b.y);
  return Circle{cx, cy, std::sqrt(dist2_xy(a.x, a.y, b.x, b.y)) * 0.5, 2, true};
}

static Circle circle_three(const Point& a, const Point& b, const Point& c, double eps) {
  const double d = 2.0 * (a.x * (b.y - c.y) +
                          b.x * (c.y - a.y) +
                          c.x * (a.y - b.y));
  const double scale = 1.0 + std::fabs(a.x) + std::fabs(a.y) +
                       std::fabs(b.x) + std::fabs(b.y) +
                       std::fabs(c.x) + std::fabs(c.y);
  // Nearly collinear triples do not define a stable circumcircle. They are
  // skipped here; their minimal enclosing circle is covered by a two-point
  // diameter candidate during pair enumeration.
  if (std::fabs(d) <= eps * scale * scale) {
    return Circle{0.0, 0.0, 0.0, 3, false};
  }

  const double aa = a.x * a.x + a.y * a.y;
  const double bb = b.x * b.x + b.y * b.y;
  const double cc = c.x * c.x + c.y * c.y;

  const double ux = (aa * (b.y - c.y) +
                     bb * (c.y - a.y) +
                     cc * (a.y - b.y)) / d;
  const double uy = (aa * (c.x - b.x) +
                     bb * (a.x - c.x) +
                     cc * (b.x - a.x)) / d;
  return Circle{ux, uy, std::sqrt(dist2_xy(ux, uy, a.x, a.y)), 3, true};
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

// A static KD-tree over the unique weighted locations. It supports:
//   - k-nearest-neighbor queries for a fast initial upper bound,
//   - disk range queries for the final exact candidate set,
//   - weighted disk sums for testing candidate support circles.
//
// Each node stores a bounding box and subtree weight. The subtree weight lets
// range_weight add an entire subtree in O(1) when its bounding box is fully
// contained in the query circle.
class KDTree {
public:
  explicit KDTree(const std::vector<Point>& points) : pts_(&points), root_(-1) {
    ids_.resize(points.size());
    std::iota(ids_.begin(), ids_.end(), 0);
    nodes_.reserve(points.size());
    root_ = build(0, static_cast<int>(ids_.size()), 0);
  }

  void range_indices(double x, double y, double r, std::vector<int>& out, double eps) const {
    out.clear();
    const double r2 = r * r;
    range_indices_rec(root_, x, y, r2, eps, out);
  }

  long long range_weight(double x, double y, double r, double eps) const {
    const double r2 = r * r;
    return range_weight_rec(root_, x, y, r2, eps);
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
    // Minimum squared distance from a point to an axis-aligned box. If this is
    // larger than the query radius squared, the subtree cannot contribute.
    double dx = 0.0;
    if (x < n.minx) dx = n.minx - x;
    else if (x > n.maxx) dx = x - n.maxx;

    double dy = 0.0;
    if (y < n.miny) dy = n.miny - y;
    else if (y > n.maxy) dy = y - n.maxy;

    return dx * dx + dy * dy;
  }

  static double bbox_max_dist2(const KDNode& n, double x, double y) {
    // Maximum squared distance from a point to an axis-aligned box. If this is
    // within the query radius squared, the whole subtree lies inside the disk.
    const double dx = std::max(std::fabs(x - n.minx), std::fabs(x - n.maxx));
    const double dy = std::max(std::fabs(y - n.miny), std::fabs(y - n.maxy));
    return dx * dx + dy * dy;
  }

  void range_indices_rec(int node_id, double x, double y, double r2,
                         double eps, std::vector<int>& out) const {
    if (node_id < 0) return;
    const KDNode& n = nodes_[node_id];
    if (!leq_dist2(bbox_min_dist2(n, x, y), r2, eps)) return;

    const Point& p = (*pts_)[n.idx];
    if (leq_dist2(dist2_xy(x, y, p.x, p.y), r2, eps)) out.push_back(n.idx);
    range_indices_rec(n.left, x, y, r2, eps, out);
    range_indices_rec(n.right, x, y, r2, eps, out);
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

struct SupportKey {
  int a;
  int b;
  int c;
  int size;

  static SupportKey one(int x) {
    return SupportKey{x, -1, -1, 1};
  }

  static SupportKey two(int x, int y) {
    if (y < x) std::swap(x, y);
    return SupportKey{x, y, -1, 2};
  }

  static SupportKey three(int x, int y, int z) {
    if (y < x) std::swap(x, y);
    if (z < y) std::swap(y, z);
    if (y < x) std::swap(x, y);
    return SupportKey{x, y, z, 3};
  }

  bool operator==(const SupportKey& other) const {
    return a == other.a && b == other.b && c == other.c && size == other.size;
  }
};

// Canonical support keys are used to cache disk weights for circles generated
// by the same 1-, 2-, or 3-point support set. Reusing an exact previous weight
// never changes the search space; it only avoids a repeated KD-tree sum.
struct SupportKeyHash {
  std::size_t operator()(const SupportKey& key) const {
    std::size_t h = static_cast<std::size_t>(key.size);
    h ^= static_cast<std::size_t>(key.a + 0x9e3779b9) + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(key.b + 0x9e3779b9) + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(key.c + 0x9e3779b9) + (h << 6) + (h >> 2);
    return h;
  }
};

struct NeighborhoodEntry {
  double x;
  double y;
  double radius;
  std::vector<int> ids;
};

class SharedCaches {
public:
  SharedCaches(bool use_weight_cache, bool use_neighborhood_cache)
      : use_weight_cache_(use_weight_cache),
        use_neighborhood_cache_(use_neighborhood_cache) {}

  long long circle_weight(const SupportKey& key,
                          const Circle& circle,
                          const KDTree& tree,
                          double eps) {
    if (!use_weight_cache_) {
      return tree.range_weight(circle.x, circle.y, circle.r, eps);
    }

    {
      std::lock_guard<std::mutex> lock(weight_mutex_);
      auto it = weight_cache_.find(key);
      if (it != weight_cache_.end()) return it->second;
    }

    const long long value = tree.range_weight(circle.x, circle.y, circle.r, eps);
    {
      std::lock_guard<std::mutex> lock(weight_mutex_);
      auto inserted = weight_cache_.emplace(key, value);
      return inserted.first->second;
    }
  }

  void range_indices(double x,
                     double y,
                     double radius,
                     const KDTree& tree,
                     double eps,
                     std::vector<int>& out) {
    if (!use_neighborhood_cache_) {
      tree.range_indices(x, y, radius, out, eps);
      return;
    }

    std::vector<int> superset;
    bool found = false;
    {
      std::lock_guard<std::mutex> lock(neighborhood_mutex_);
      for (const NeighborhoodEntry& entry : neighborhoods_) {
        const double center_dist = std::sqrt(dist2_xy(x, y, entry.x, entry.y));
        const double margin = eps * (1.0 + entry.radius + radius + center_dist);
        // Reuse a previous neighborhood only when the new disk is provably
        // contained in it. We then filter the stored superset exactly.
        if (center_dist + radius <= entry.radius - margin) {
          superset = entry.ids;
          found = true;
          break;
        }
      }
    }

    if (found) {
      out.clear();
      out.reserve(superset.size());
      const double r2 = radius * radius;
      for (int id : superset) {
        const Point& p = points_for_filter_->at(id);
        if (leq_dist2(dist2_xy(x, y, p.x, p.y), r2, eps)) out.push_back(id);
      }
      return;
    }

    tree.range_indices(x, y, radius, out, eps);
    {
      std::lock_guard<std::mutex> lock(neighborhood_mutex_);
      neighborhoods_.push_back(NeighborhoodEntry{x, y, radius, out});
    }
  }

  void set_points_for_filter(const std::vector<Point>* points) {
    points_for_filter_ = points;
  }

private:
  bool use_weight_cache_;
  bool use_neighborhood_cache_;
  const std::vector<Point>* points_for_filter_ = nullptr;
  std::mutex weight_mutex_;
  std::mutex neighborhood_mutex_;
  std::unordered_map<SupportKey, long long, SupportKeyHash> weight_cache_;
  std::vector<NeighborhoodEntry> neighborhoods_;
};

static bool contains_all(const Circle& c, const std::vector<int>& ids,
                         const std::vector<Point>& pts, double eps) {
  for (int id : ids) {
    if (!point_in_circle(pts[id], c, eps)) return false;
  }
  return true;
}

// Minimal enclosing circle for a small seed set. This is used only to obtain
// a valid upper bound R. The final answer is still certified later by complete
// support enumeration inside the exact 2R candidate region.
static Circle mec_bruteforce(const std::vector<int>& ids,
                             const std::vector<Point>& pts,
                             double eps) {
  Circle best{0.0, 0.0, std::numeric_limits<double>::infinity(), 0, false};
  if (ids.empty()) return best;

  for (int a : ids) {
    Circle c = circle_one(pts[a]);
    if (c.r <= best.r && contains_all(c, ids, pts, eps)) best = c;
  }

  for (std::size_t i = 0; i < ids.size(); ++i) {
    for (std::size_t j = i + 1; j < ids.size(); ++j) {
      Circle c = circle_two(pts[ids[i]], pts[ids[j]]);
      if (c.r < best.r && contains_all(c, ids, pts, eps)) best = c;
    }
  }

  for (std::size_t i = 0; i < ids.size(); ++i) {
    for (std::size_t j = i + 1; j < ids.size(); ++j) {
      for (std::size_t h = j + 1; h < ids.size(); ++h) {
        Circle c = circle_three(pts[ids[i]], pts[ids[j]], pts[ids[h]], eps);
        if (c.ok && c.r < best.r && contains_all(c, ids, pts, eps)) best = c;
      }
    }
  }
  return best;
}

struct SolveResult {
  double x;
  double y;
  double r;
  double weight;
  int support;
  int candidates;
};

static void maybe_update(const Circle& c, int anchor,
                         const SupportKey& support_key,
                         const std::vector<Point>& pts,
                         const KDTree& tree,
                         SharedCaches& caches,
                         int k,
                         double eps,
                         Circle& best,
                         long long& best_weight) {
  if (!c.ok) return;
  // We only care about circles that improve the current upper bound, contain
  // the required anchor, and have enough weighted input points inside.
  if (c.r >= best.r - eps * (1.0 + best.r)) return;
  if (!point_in_circle(pts[anchor], c, eps)) return;

  const long long w = caches.circle_weight(support_key, c, tree, eps);
  if (w >= k) {
    best = c;
    best_weight = w;
  }
}

static void enumerate_supports(const std::vector<int>& cand,
                               int anchor,
                               const std::vector<Point>& pts,
                               const KDTree& tree,
                               SharedCaches& caches,
                               int k,
                               double eps,
                               Circle& best,
                               long long& best_weight) {
  const std::size_t m = cand.size();

  // Enumerating all supports of size 1, 2, and 3 is exact in R^2: every
  // smallest feasible circle has a minimal enclosing circle whose boundary is
  // determined by at most three points.
  for (std::size_t i = 0; i < m; ++i) {
    maybe_update(circle_one(pts[cand[i]]), anchor, SupportKey::one(cand[i]),
                 pts, tree, caches, k, eps, best, best_weight);
  }

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = i + 1; j < m; ++j) {
      const double d2 = dist2(pts[cand[i]], pts[cand[j]]);
      // A circle better than best cannot contain two support points farther
      // apart than its diameter.
      if (d2 >= 4.0 * best.r * best.r) continue;
      maybe_update(circle_two(pts[cand[i]], pts[cand[j]]),
                   anchor, SupportKey::two(cand[i], cand[j]),
                   pts, tree, caches, k, eps, best, best_weight);
    }
  }

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = i + 1; j < m; ++j) {
      const double dij = dist2(pts[cand[i]], pts[cand[j]]);
      if (dij >= 4.0 * best.r * best.r) continue;

      for (std::size_t h = j + 1; h < m; ++h) {
        const double dih = dist2(pts[cand[i]], pts[cand[h]]);
        // The same diameter bound prunes triples before constructing their
        // circumcircle. This pruning is exact.
        if (dih >= 4.0 * best.r * best.r) continue;
        const double djh = dist2(pts[cand[j]], pts[cand[h]]);
        if (djh >= 4.0 * best.r * best.r) continue;

        Circle c = circle_three(pts[cand[i]], pts[cand[j]], pts[cand[h]], eps);
        if (!c.ok || c.r >= best.r - eps * (1.0 + best.r)) continue;
        maybe_update(c, anchor, SupportKey::three(cand[i], cand[j], cand[h]),
                     pts, tree, caches, k, eps, best, best_weight);
      }
    }
  }
}

static SolveResult solve_one(int anchor,
                             const std::vector<Point>& pts,
                             const KDTree& tree,
                             SharedCaches& caches,
                             int k,
                             int seed_size,
                             double eps) {
  const int n = static_cast<int>(pts.size());
  if (pts[anchor].w >= k) {
    // If the anchor location itself has enough multiplicity, radius zero is
    // optimal and no spatial search is needed.
    return SolveResult{pts[anchor].x, pts[anchor].y, 0.0,
                       static_cast<double>(pts[anchor].w), 1, 1};
  }

  // First collect nearest unique locations until their total positive weight
  // reaches k. For ordinary anchors this is usually just k nearest locations;
  // for a temporary zero-weight anchor t outside U, it may require one more.
  int init_k = std::min(n, std::max(1, k));
  std::vector<int> nn;
  std::vector<int> selected;
  long long selected_weight = 0;

  while (true) {
    nn = tree.knearest(pts[anchor].x, pts[anchor].y, init_k, eps);
    selected.clear();
    selected.reserve(init_k);
    selected_weight = 0;
    for (int id : nn) {
      selected.push_back(id);
      selected_weight += pts[id].w;
      if (selected_weight >= k) break;
    }
    if (selected_weight >= k || init_k == n) break;
    init_k = std::min(n, std::max(init_k + 1, init_k * 2));
  }

  Circle best = mec_bruteforce(selected, pts, eps);
  long long best_weight = tree.range_weight(best.x, best.y, best.r, eps);

  // A larger local seed can improve the upper bound before the exact phase.
  // This is a heuristic speedup only: correctness does not depend on it.
  const int seed_k = std::min(n, std::max(seed_size, init_k));
  if (seed_k > init_k) {
    std::vector<int> seed = tree.knearest(pts[anchor].x, pts[anchor].y, seed_k, eps);
    enumerate_supports(seed, anchor, pts, tree, caches, k, eps, best, best_weight);
  }

  std::vector<int> cand;
  // Exact candidate reduction:
  // If a feasible circle of radius R contains the anchor, then every point in
  // any strictly better feasible circle has distance at most 2R from the anchor.
  // Therefore all supports of the optimum are inside this disk.
  caches.range_indices(pts[anchor].x, pts[anchor].y, 2.0 * best.r, tree, eps, cand);
  enumerate_supports(cand, anchor, pts, tree, caches, k, eps, best, best_weight);

  return SolveResult{best.x, best.y, best.r,
                     static_cast<double>(best_weight), best.support,
                     static_cast<int>(cand.size())};
}

class CircleWorker : public RcppParallel::Worker {
public:
  CircleWorker(const std::vector<Point>& pts,
               const KDTree& tree,
               SharedCaches& caches,
               int k,
               int seed_size,
               double eps,
               NumericVector cx,
               NumericVector cy,
               NumericVector radius,
               NumericVector out_weight,
               IntegerVector support,
               IntegerVector candidates)
      : pts_(pts), tree_(tree), caches_(caches),
        k_(k), seed_size_(seed_size), eps_(eps),
        cx_(cx), cy_(cy), radius_(radius), out_weight_(out_weight),
        support_(support), candidates_(candidates) {}

  void operator()(std::size_t begin, std::size_t end) {
    // Each unique anchor can be solved independently. Shared caches are
    // protected internally; a missed cache hit only costs time, not correctness.
    for (std::size_t i = begin; i < end; ++i) {
      SolveResult ans = solve_one(static_cast<int>(i), pts_, tree_, caches_,
                                   k_, seed_size_, eps_);
      cx_[i] = ans.x;
      cy_[i] = ans.y;
      radius_[i] = ans.r;
      out_weight_[i] = ans.weight;
      support_[i] = ans.support;
      candidates_[i] = ans.candidates;
    }
  }

private:
  const std::vector<Point>& pts_;
  const KDTree& tree_;
  SharedCaches& caches_;
  int k_;
  int seed_size_;
  double eps_;
  RcppParallel::RVector<double> cx_;
  RcppParallel::RVector<double> cy_;
  RcppParallel::RVector<double> radius_;
  RcppParallel::RVector<double> out_weight_;
  RcppParallel::RVector<int> support_;
  RcppParallel::RVector<int> candidates_;
};

// U is the original multiset: every row is one occurrence of a location.
// Equal coordinate pairs are aggregated internally into U* with integer weights.
// The heuristic seed only tightens the upper bound. The final support enumeration
// over all points within 2R of the anchor is the exact step.
//
// [[Rcpp::export]]
Rcpp::DataFrame weighted_anchor_circles(const NumericMatrix& U,
                                        int k,
                                        int seed_size = 32,
                                        bool parallel = true,
                                        bool use_weight_cache = true,
                                        bool use_neighborhood_cache = true,
                                        double eps = 1e-10) {
  if (U.ncol() != 2) Rcpp::stop("U must have exactly two columns.");
  if (k < 1) Rcpp::stop("k must be >= 1.");
  if (seed_size < k) seed_size = k;

  const int n = U.nrow();
  if (n == 0) Rcpp::stop("U must contain at least one row.");
  if (n < k) Rcpp::stop("nrow(U) must be >= k.");

  std::vector<Point> pts;
  pts.reserve(n);
  std::vector<int> original_to_unique(n);
  std::map<std::pair<double, double>, int> id_by_coord;

  // Compress the multiset U into U*. Exact equality of the two double
  // coordinates defines equality of locations here. If coordinates should be
  // snapped or rounded, do that before calling this function.
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
  SharedCaches caches(use_weight_cache, use_neighborhood_cache);
  caches.set_points_for_filter(&pts);

  // Solve once per unique location, then expand the result back to every
  // original row of the multiset. Duplicate input rows therefore receive
  // duplicate output rows with the same circle.
  const int m = static_cast<int>(pts.size());
  NumericVector unique_cx(m), unique_cy(m), unique_radius(m), unique_weight(m);
  IntegerVector support(m), candidates(m);
  CircleWorker worker(pts, tree, caches, k, seed_size, eps,
                      unique_cx, unique_cy, unique_radius, unique_weight,
                      support, candidates);

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
    cx[i] = unique_cx[id];
    cy[i] = unique_cy[id];
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

// Compute the exact circle for one requested location t. If t is present in U,
// its multiplicity contributes to the weight. Otherwise t is only the required
// anchor point and has weight zero.
//
// [[Rcpp::export]]
Rcpp::DataFrame weighted_anchor_circle_at(const NumericMatrix& U,
                                          const NumericVector& t,
                                          int k,
                                          int seed_size = 32,
                                          bool use_weight_cache = true,
                                          bool use_neighborhood_cache = true,
                                          double eps = 1e-10) {
  if (U.ncol() != 2) Rcpp::stop("U must have exactly two columns.");
  if (t.size() != 2) Rcpp::stop("t must have length 2.");
  if (!R_finite(t[0]) || !R_finite(t[1])) Rcpp::stop("t contains non-finite coordinates.");
  if (k < 1) Rcpp::stop("k must be >= 1.");
  if (seed_size < k) seed_size = k;

  const int n = U.nrow();
  if (n == 0) Rcpp::stop("U must contain at least one row.");
  if (n < k) Rcpp::stop("nrow(U) must be >= k.");

  std::vector<Point> pts;
  pts.reserve(n + 1);
  std::map<std::pair<double, double>, int> id_by_coord;

  // Build the same weighted unique-location representation as the all-anchor
  // function. No output expansion is needed because only one target is solved.
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
      pts.push_back(Point{x, y, 1});
    } else {
      pts[it->second].w += 1;
    }
  }

  const std::pair<double, double> target_key(t[0], t[1]);
  auto target_it = id_by_coord.find(target_key);
  int anchor = -1;
  if (target_it == id_by_coord.end()) {
    // t is not an input location. Add it as a required anchor with weight zero:
    // it constrains the circle but does not contribute to the k weight.
    anchor = static_cast<int>(pts.size());
    pts.push_back(Point{t[0], t[1], 0});
  } else {
    // t is an input location. Its existing multiplicity is already stored as
    // the point weight and therefore contributes to the weight threshold.
    anchor = target_it->second;
  }

  KDTree tree(pts);
  SharedCaches caches(use_weight_cache, use_neighborhood_cache);
  caches.set_points_for_filter(&pts);

  SolveResult ans = solve_one(anchor, pts, tree, caches, k, seed_size, eps);

  return Rcpp::DataFrame::create(
    Rcpp::Named("px") = NumericVector::create(t[0]),
    Rcpp::Named("py") = NumericVector::create(t[1]),
    Rcpp::Named("cx") = NumericVector::create(ans.x),
    Rcpp::Named("cy") = NumericVector::create(ans.y),
    Rcpp::Named("radius") = NumericVector::create(ans.r),
    Rcpp::Named("weight") = NumericVector::create(ans.weight)
  );
}
