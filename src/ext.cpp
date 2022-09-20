#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include <cstring>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <omp.h>

namespace nb = nanobind;

using namespace nb::literals;

template <typename T>
struct uniform_axis_t {
  int64_t nbins;
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
  T x_i;
  for (size_t i = 0; i < x.shape(0); ++i) {
    x_i = x(i);
    if (x_i < axis.amin) continue;
    if (x_i >= axis.amax) continue;
    bin = static_cast<int64_t>((x_i - axis.amin) * norm);
    output[bin]++;
  }
}

template <typename Tx, typename Te>
int64_t calc_bin(Tx x, const std::vector<Te>& edges) {
  auto s = std::distance(std::begin(edges),
                         std::lower_bound(std::begin(edges), std::end(edges), x));
  return static_cast<int64_t>(s) - 1;
}

template <typename Tx, typename Te>
void _v1d(int64_t* output,
          nb::tensor<Tx, nb::shape<nb::any>> x,
          const std::vector<Te>& edges) {
  int64_t nbins = edges.size() - 1;
  auto amin = edges.front();
  auto amax = edges.back();
  auto nx = x.shape(0);
#pragma omp parallel
  {
    std::vector<int64_t> values_ot(nbins, 0);
    int64_t bin;
    Tx x_i;
#pragma omp for nowait
    for (int64_t i = 0; i < nx; ++i) {
      x_i = x(i);
      if (x_i < amin) continue;
      if (x_i >= amax) continue;
      bin = calc_bin(x_i, edges);
      values_ot[bin]++;
    }
#pragma omp critical
    for (int64_t i = 0; i < nbins; ++i) {
      output[i] += values_ot[i];
    }
  }
}



template <typename T>
T* zeros_1d(int64_t size) {
  auto data = new T[size];
  std::memset(data, 0, size * sizeof(T));
  return data;
}

template <typename T>
nb::tensor<nb::numpy, int64_t, nb::shape<nb::any>> f1d(nb::tensor<T, nb::shape<nb::any>> x,
                                                       int64_t nbins,
                                                       T xmin,
                                                       T xmax) {
  auto data = zeros_1d<int64_t>(nbins);
  size_t shape[1] = {static_cast<size_t>(nbins)};
  _f1d(data, x, {.nbins = nbins, .amax = xmax, .amin = xmin});
  nb::capsule owner(
    data,
    [](void *p) noexcept { delete[] (int64_t*)p; }
  );
  return nb::tensor<nb::numpy, int64_t, nb::shape<nb::any>>(data, 1, shape, owner);
}


template <typename T>
nb::tensor<nb::numpy, int64_t, nb::shape<nb::any>> v1d(nb::tensor<T, nb::shape<nb::any>> x,
                                                       nb::tensor<double, nb::shape<nb::any>> edges) {
  auto nedges =  edges.shape(0);
  auto nbins = nedges - 1;
  std::vector<double> edges_v(reinterpret_cast<double*>(edges.data()),
                              reinterpret_cast<double*>(edges.data()) + nedges);

  auto data = zeros_1d<int64_t>(nbins);
  size_t shape[1] = {static_cast<size_t>(nbins)};
  _v1d(data, x, edges_v);
  nb::capsule owner(
    data,
    [](void *p) noexcept { delete[] (int64_t*)p; }
  );
  return nb::tensor<nb::numpy, int64_t, nb::shape<nb::any>>(data, 1, shape, owner);
}

NB_MODULE(ext, m) {
  m.def("f1d", &f1d<double>, "x", "nbins", "xmin", "xmax");
  m.def("v1d", &v1d<double>, "x", "edges");
}
