#include <cuda_runtime_api.h>

namespace detrex {
int get_cudart_version() {
  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);
  return runtimeVersion;
}
} // namespace detrex
