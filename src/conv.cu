// Dense NC[D]HW convolution through the cuDNN v8 graph API, selecting engines exactly like torch 2.6 Conv_v8.cpp with
// benchmark=False: INSTANT heuristics, drop DOWN_CONVERT_INPUTS engines, first plan that builds wins; bias is a
// separate broadcast add (as in at::convolution).
#include "ops.h"
#include <cudnn_frontend.h>

#define CD(x) do { auto s_ = (x); REQ(s_ == CUDNN_STATUS_SUCCESS, "cudnn %s", cudnnGetErrorString(s_)); } while (0)

static cudnnHandle_t hdl() {
  static cudnnHandle_t h = [] { cudnnHandle_t h; CD(cudnnCreate(&h)); CD(cudnnSetStream(h, stream())); return h; }();
  return h;
}
static uint8_t alignment(const void* p) { uint8_t a = 1; for (; a < 32; a *= 2) if ((uintptr_t)p % (a * 2)) return a; return a; }
static cudnnDataType_t cdt(DT d) { return d == F16 ? CUDNN_DATA_HALF : d == BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT; }

static cudnn_frontend::Tensor tdesc(const std::vector<i64>& sh, int64_t id, uint8_t al, DT dt) {
  int n = (int)sh.size(); std::vector<int64_t> s(n), st(n);
  for (int i = 0; i < n; i++) s[i] = sh[i];
  st[n - 1] = 1; for (int i = n - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1];
  std::vector<int> perm; for (int d = n - 1; d > 1; d--) perm.push_back(d); perm.push_back(1); perm.push_back(0);
  int64_t z = 1; for (int d : perm) { if (s[d] == 1) st[d] = z; else z *= s[d]; }  // fixSizeOneDimStride (NCDHW)
  return cudnn_frontend::TensorBuilder().setDim(n, s.data()).setStrides(n, st.data()).setId(id).setAlignment(al).setDataType(cdt(dt)).build();
}

struct ConvKey {
  i64 x[5], w[5]; int nd, pad, stride, dt; uint8_t ax, ay, aw;
  bool operator<(const ConvKey& o) const { return memcmp(this, &o, sizeof(*this)) < 0; }
};

void conv(const Tensor& x, const Tensor& w, const Tensor* b, int pad, int stride, Tensor& y) {
  static std::map<ConvKey, std::shared_ptr<cudnn_frontend::ExecutionPlan>> cache;
  int nd = x.dim() - 2; REQ(nd == 2 || nd == 3, "conv dims");
  ConvKey key{}; for (int i = 0; i < nd + 2; i++) { key.x[i] = x.sh[i]; key.w[i] = w.sh[i]; }
  key.nd = nd; key.pad = pad; key.stride = stride; key.dt = x.dt; key.ax = alignment(x.p); key.ay = alignment(y.p); key.aw = alignment(w.p);
  auto it = cache.find(key);
  if (it == cache.end()) {
    int64_t pd[3] = {pad, pad, pad}, sd[3] = {stride, stride, stride}, dl[3] = {1, 1, 1};
    auto conv = cudnn_frontend::ConvDescBuilder().setDataType(CUDNN_DATA_FLOAT).setMathMode(CUDNN_CROSS_CORRELATION).setNDims(nd)
                    .setStrides(nd, sd).setPrePadding(nd, pd).setPostPadding(nd, pd).setDilation(nd, dl).build();
    auto op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                  .setxDesc(tdesc(x.sh, 'x', key.ax, x.dt)).setyDesc(tdesc(y.sh, 'y', key.ay, y.dt))
                  .setwDesc(tdesc(w.sh, 'w', key.aw, w.dt)).setcDesc(conv).build();
    std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
    auto graph = cudnn_frontend::OperationGraphBuilder().setHandle(hdl()).setOperationGraph(ops.size(), ops.data()).build();
    auto tag = graph.getTag();
    auto try_list = [&](cudnn_frontend::EngineConfigList& cfgs) -> std::shared_ptr<cudnn_frontend::ExecutionPlan> {
      cudnn_frontend::EngineConfigList keep;
      cudnn_frontend::filter(cfgs, keep, [](cudnnBackendDescriptor_t c) {
        return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c);
      });
      for (auto& c : keep) {
        try {
          auto p = std::make_shared<cudnn_frontend::ExecutionPlan>(cudnn_frontend::ExecutionPlanBuilder().setHandle(hdl()).setEngineConfig(c, tag).build());
          return p;
        } catch (cudnn_frontend::cudnnException&) {}
      }
      return nullptr;
    };
    auto heur = cudnn_frontend::EngineHeuristicsBuilder().setOperationGraph(graph).setHeurMode(CUDNN_HEUR_MODE_INSTANT).build();
    auto& hc = heur.getEngineConfig(heur.getEngineConfigCount());
    auto plan = try_list(hc);
    if (!plan) {
      auto fb = cudnn_frontend::EngineFallbackListBuilder().setOperationGraph(graph).setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR).build();
      plan = try_list(fb.getFallbackList());
    }
    REQ(plan, "no cudnn engine");
    if (env("T2_VERBOSE") == "1") fprintf(stderr, "[conv] %s\n", plan->getTag().c_str());
    it = cache.emplace(key, plan).first;
  }
  auto& plan = *it->second;
  Tensor ws = empty({(i64)plan.getWorkspaceSize()}, U8);
  void* ptrs[3] = {x.p, y.p, w.p}; int64_t uids[3] = {'x', 'y', 'w'};
  auto vp = cudnn_frontend::VariantPackBuilder().setWorkspacePointer(plan.getWorkspaceSize() ? ws.p : nullptr).setDataPointers(3, ptrs).setUids(3, uids).build();
  CD(cudnnBackendExecute(hdl(), plan.get_raw_desc(), vp.get_raw_desc()));
  if (b) bias_add_ncdhw_(y, *b);
}
Tensor conv(const Tensor& x, const Tensor& w, const Tensor* b, int pad, int stride) {
  std::vector<i64> sh = {x.size(0), w.size(0)};
  for (int i = 2; i < x.dim(); i++) sh.push_back((x.size(i) + 2 * pad - w.size(i)) / stride + 1);
  Tensor y = empty(sh, x.dt);
  conv(x, w, b, pad, stride, y);
  return y;
}
