#include <torch/extension.h>
#include "qpf_rasterizer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qpf_render_forward",  &qpf_2d_cuda::qpf_render_forward,  "QPF 2D gaussian render forward (linear coherent sum)");
    m.def("qpf_render_backward", &qpf_2d_cuda::qpf_render_backward, "QPF 2D gaussian render backward");
}
