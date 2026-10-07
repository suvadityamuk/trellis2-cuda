#pragma once

#include <torch/torch.h>

#include <vector>

std::vector<torch::Tensor> mesh_udf(torch::Tensor const& vertices, torch::Tensor const& faces, torch::Tensor const& points);
torch::Tensor telea_inpaint(torch::Tensor const& image, torch::Tensor const& mask, int64_t radius);
