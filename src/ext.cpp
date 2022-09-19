#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include <cstring>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace nb = nanobind;

using namespace nb::literals;

template <typename T>
struct uniform_axis_t {
  uniform_axis_t(size_t nb, T ami, T ama) {
    nbins = nb;
    amin = ami;
    amax = ama;
  };
  size_t nbins;
  T amin;
  T amax;
};

template <typename T>
T uanorm(uniform_axis_t<T> ax) {
  return ax.nbins / (ax.amax - ax.amin);
}

template <typename T>
void _f1d(int64_t* output,
          nb::tensor<T, nb::shape<nb::any>> x, uniform_axis_t<T> axis) {
  T norm = uanorm(axis);
  int64_t bin;
  for (size_t i = 0; i < x.shape(0); ++i) {
    T xi = x(i);
    if (xi < axis.amin) continue;
    if (xi >= axis.amax) continue;
    bin = static_cast<int64_t>((xi - axis.amin) * norm);
    output[bin]++;
  }
}

template <typename T>
nb::tensor<nb::numpy, int64_t, nb::shape<nb::any>> f1d(
                                                       nb::tensor<T, nb::shape<nb::any>> x, int64_t nbins, T xmin, T xmax) {
  int64_t* data = new int64_t[nbins];
  std::memset(data, 0, nbins * sizeof(int64_t));
  size_t shape[1] = {static_cast<size_t>(nbins)};
  auto axis = uniform_axis_t(nbins, xmin, xmax);
  _f1d(data, x, axis);
  nb::capsule owner(data, [](void *p) noexcept {
    delete[] (int64_t *) p;
  });
  return nb::tensor<nb::numpy, int64_t, nb::shape<nb::any>>(data, 1, shape, owner);
}

NB_MODULE(ext, m) {
  m.def("f1d", &f1d<double>, "x", "nbins", "xmin", "xmax");
}
