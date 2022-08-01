#include <cuda_runtime_api.h>

namespace idet {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace idet
