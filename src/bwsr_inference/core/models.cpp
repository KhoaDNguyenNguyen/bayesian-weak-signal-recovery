#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace ssr_inference {
namespace core {

extern void evaluate_gaussian(const double*, std::size_t, double, double, double, double*);
extern void evaluate_pseudo_voigt(const double*, std::size_t, double, double, double, double, double*);
extern void evaluate_exponential_background(const double*, std::size_t, double, double, double, double*);
extern void evaluate_polynomial_background(const double*, std::size_t, const double*, std::size_t, double*);

py::array_t<double> py_evaluate_gaussian(py::array_t<double> f, double amplitude, double mu, double sigma) {
    py::buffer_info buf = f.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Independent variable array must be 1-dimensional.");
    }

    auto result = py::array_t<double>(buf.size);
    py::buffer_info res_buf = result.request();

    evaluate_gaussian(static_cast<const double*>(buf.ptr), buf.size, amplitude, mu, sigma, static_cast<double*>(res_buf.ptr));
    return result;
}

py::array_t<double> py_evaluate_pseudo_voigt(py::array_t<double> f, double amplitude, double mu, double sigma, double gamma) {
    py::buffer_info buf = f.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Independent variable array must be 1-dimensional.");
    }

    auto result = py::array_t<double>(buf.size);
    py::buffer_info res_buf = result.request();

    evaluate_pseudo_voigt(static_cast<const double*>(buf.ptr), buf.size, amplitude, mu, sigma, gamma, static_cast<double*>(res_buf.ptr));
    return result;
}

py::array_t<double> py_evaluate_exponential_background(py::array_t<double> f, double amplitude, double decay_constant, double offset) {
    py::buffer_info buf = f.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Independent variable array must be 1-dimensional.");
    }

    auto result = py::array_t<double>(buf.size);
    py::buffer_info res_buf = result.request();

    evaluate_exponential_background(static_cast<const double*>(buf.ptr), buf.size, amplitude, decay_constant, offset, static_cast<double*>(res_buf.ptr));
    return result;
}

py::array_t<double> py_evaluate_polynomial_background(py::array_t<double> f, py::array_t<double> coeffs) {
    py::buffer_info buf_f = f.request();
    py::buffer_info buf_c = coeffs.request();

    if (buf_f.ndim != 1 || buf_c.ndim != 1) {
        throw std::runtime_error("Independent variable and coefficients arrays must be 1-dimensional.");
    }
    
    if (buf_c.size == 0) {
        throw std::invalid_argument("Coefficient array must not be empty.");
    }

    auto result = py::array_t<double>(buf_f.size);
    py::buffer_info res_buf = result.request();

    std::size_t degree = buf_c.size - 1;

    evaluate_polynomial_background(static_cast<const double*>(buf_f.ptr), buf_f.size, static_cast<const double*>(buf_c.ptr), degree, static_cast<double*>(res_buf.ptr));
    return result;
}

PYBIND11_MODULE(_core_models, m) {
    m.doc() = "C++ Core bindings for ssr_inference parametric models.";

    m.def("evaluate_gaussian", &py_evaluate_gaussian, 
          "Evaluate Gaussian transient signal.",
          py::arg("f"), py::arg("amplitude"), py::arg("mu"), py::arg("sigma"));
          
    m.def("evaluate_pseudo_voigt", &py_evaluate_pseudo_voigt, 
          "Evaluate Pseudo-Voigt transient signal.",
          py::arg("f"), py::arg("amplitude"), py::arg("mu"), py::arg("sigma"), py::arg("gamma"));
          
    m.def("evaluate_exponential_background", &py_evaluate_exponential_background, 
          "Evaluate exponential decay background.",
          py::arg("f"), py::arg("amplitude"), py::arg("decay_constant"), py::arg("offset"));
          
    m.def("evaluate_polynomial_background", &py_evaluate_polynomial_background, 
          "Evaluate polynomial background.",
          py::arg("f"), py::arg("coeffs"));
}

} // namespace core
} // namespace ssr_inference