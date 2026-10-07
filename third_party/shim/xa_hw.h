// Thread count for xatlas: hardware_concurrency() ignores the cgroup CPU quota (192 vs 23 on HF Jobs).
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <thread>
static inline unsigned xa_hw() {
  static unsigned n = [] {
    if (const char* e = std::getenv("XA_THREADS")) return (unsigned)std::max(1, std::atoi(e));
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    long q = 0, p = 0;
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
      if (std::fscanf(f, "%ld %ld", &q, &p) != 2) q = 0;
      std::fclose(f);
    }
    return q > 0 && p > 0 ? std::min(hw, (unsigned)((q + p - 1) / p)) : hw;
  }();
  return n;
}
