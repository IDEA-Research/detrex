#include <cuda_runtime_api.h>

namespace detrex {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace detrex
