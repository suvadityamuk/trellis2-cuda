#pragma once
#include <cstdio>
#include <cstdlib>
#define TORCH_CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "TORCH_CHECK failed: %s @ %s:%d\n", #c, __FILE__, __LINE__); abort(); } } while (0)
