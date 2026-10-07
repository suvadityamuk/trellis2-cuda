#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("mesh_udf(Tensor vertices, Tensor faces, Tensor points) -> Tensor[]");
  ops.def("telea_inpaint(Tensor image, Tensor mask, int radius) -> Tensor");
#if defined(CUDA_KERNEL)
  ops.impl("mesh_udf", torch::kCUDA, &mesh_udf);
  ops.impl("telea_inpaint", torch::kCUDA, &telea_inpaint);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
