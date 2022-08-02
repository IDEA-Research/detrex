#include <cuda_runtime_api.h>

namespace ideadet {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace ideadet
