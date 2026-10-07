// torch.randn on CPU, float32, numel >= 16: normal_fill_AVX2 (selected by torch 2.6 even on AVX512 hosts). mt19937
// 24-bit uniforms, Box-Muller over blocks of 16 with avx_mathfun log/sincos; tail block re-drawn when numel % 16 != 0.
#include "models.h"
#define CPU_CAPABILITY_AVX2
#include "avx_mathfun.h"

static void fill16(float* d) {
  const __m256 u1 = _mm256_sub_ps(_mm256_set1_ps(1.f), _mm256_loadu_ps(d)), u2 = _mm256_loadu_ps(d + 8);
  const __m256 r = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2.f), log256_ps(u1)));
  __m256 s, c; sincos256_ps(_mm256_mul_ps(_mm256_set1_ps((float)(2.0f * 3.14159265358979323846)), u2), &s, &c);
  _mm256_storeu_ps(d, _mm256_mul_ps(r, c)); _mm256_storeu_ps(d + 8, _mm256_mul_ps(r, s));
}
std::vector<float> torch_randn(std::mt19937& g, i64 n) {
  REQ(n >= 16, "torch_randn: n < 16 uses a different torch path");
  std::vector<float> d(n);
  auto U = [&] { return (float)(g() & 0xFFFFFF) * (1.f / 16777216.f); };
  for (auto& x : d) x = U();
  for (i64 i = 0; i + 16 <= n; i += 16) fill16(&d[i]);
  if (n % 16) { for (int i = 0; i < 16; i++) d[n - 16 + i] = U(); fill16(&d[n - 16]); }
  return d;
}
