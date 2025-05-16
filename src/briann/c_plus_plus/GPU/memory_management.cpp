#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

void store(int i) {
    return;
}

int load() {
    int i = 0;
    return i;
}

namespace py = pybind11;

PYBIND11_MODULE(GPU_memory_management_module, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           store
           load
    )pbdoc";

    m.def("store", &store, R"pbdoc(
        Stores a number.

        Some other explanation about the store function.
    )pbdoc");

    m.def("load", &load, R"pbdoc(
        Loads a number.

        Some other explanation about the load function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
