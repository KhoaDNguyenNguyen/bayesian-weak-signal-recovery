#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstddef>

namespace py = pybind11;

namespace bwsr_inference {
namespace core {

extern void evaluate_gaussian(const double* f, std::size_t size, double amplitude, double mu, double sigma, double* output);
extern void evaluate_pseudo_voigt(const double* f, std::size_t size, double amplitude, double mu, double sigma, double gamma, double* output);
extern void evaluate_exponential_background(const double* f, std::size_t size, double amplitude, double decay_constant, double offset, double* output);
extern void evaluate_polynomial_background(const double* f, std::size_t size, const double* coeffs, std::size_t degree, double* output);

/**
 * @brief Thread-safe Python wrapper for the Gaussian resonance signal evaluation.
 * 
 * @details
 * Local Global Interpreter Lock (GIL) Management Architecture:
 * The GIL is implicitly retained upon function entry to guarantee thread-safe 
 * interaction with the Python C-API (e.g., NumPy array allocation and buffer 
 * metadata extraction). The GIL is strictly released only within a localized 
 * scope block immediately prior to the execution of the computationally intensive, 
 * pure C++ numerical evaluation. This isolation mechanism facilitates true 
 * parallel execution across multi-core hardware architectures while unconditionally 
 * preventing memory segmentation faults associated with concurrent Python object access.
 * 
 * @param f The 1-dimensional NumPy array representing frequency bins.
 * @param amplitude The peak amplitude parameter.
 * @param mu The center frequency parameter.
 * @param sigma The standard deviation parameter.
 * @return A newly allocated 1-dimensional NumPy array containing the model predictions.
 */
py::array_t<double> py_evaluate_gaussian(py::array_t<double, py::array::c_style> f, double amplitude, double mu, double sigma) {
  py::buffer_info buf = f.request();

  if (buf.ndim != 1) {
    throw std::runtime_error("Independent variable array must be strictly 1-dimensional.");
  }

  // Allocate contiguous memory for the output array mapped to the Python space (Requires GIL)
  auto result = py::array_t<double>(buf.size);
  py::buffer_info res_buf = result.request();

  const double* f_ptr = static_cast<const double*>(buf.ptr);
  double* res_ptr = static_cast<double*>(res_buf.ptr);

  // High-Intensity Computational Execution Block
  {
    // Explicitly release the GIL here to facilitate true multi-threading
    pybind11::gil_scoped_release release;
    evaluate_gaussian(f_ptr, buf.size, amplitude, mu, sigma, res_ptr);
  } // The GIL is automatically re-acquired when this scope terminates

  return result;
}

/**
 * @brief Thread-safe Python wrapper for the Pseudo-Voigt resonance signal evaluation.
 * 
 * @details Utilizes identical zero-copy memory mapping and localized GIL 
 * management methodologies to guarantee thread-safe parallelism.
 */
py::array_t<double> py_evaluate_pseudo_voigt(py::array_t<double, py::array::c_style> f, double amplitude, double mu, double sigma, double gamma) {
  py::buffer_info buf = f.request();

  if (buf.ndim != 1) {
    throw std::runtime_error("Independent variable array must be strictly 1-dimensional.");
  }

  auto result = py::array_t<double>(buf.size);
  py::buffer_info res_buf = result.request();

  const double* f_ptr = static_cast<const double*>(buf.ptr);
  double* res_ptr = static_cast<double*>(res_buf.ptr);

  {
    pybind11::gil_scoped_release release;
    evaluate_pseudo_voigt(f_ptr, buf.size, amplitude, mu, sigma, gamma, res_ptr);
  }

  return result;
}

/**
 * @brief Thread-safe Python wrapper for the Exponential background model evaluation.
 * 
 * @details Utilizes identical zero-copy memory mapping and localized GIL 
 * management methodologies to guarantee thread-safe parallelism.
 */
py::array_t<double> py_evaluate_exponential_background(py::array_t<double, py::array::c_style> f, double amplitude, double decay_constant, double offset) {
  py::buffer_info buf = f.request();

  if (buf.ndim != 1) {
    throw std::runtime_error("Independent variable array must be strictly 1-dimensional.");
  }

  auto result = py::array_t<double>(buf.size);
  py::buffer_info res_buf = result.request();

  const double* f_ptr = static_cast<const double*>(buf.ptr);
  double* res_ptr = static_cast<double*>(res_buf.ptr);

  {
    pybind11::gil_scoped_release release;
    evaluate_exponential_background(f_ptr, buf.size, amplitude, decay_constant, offset, res_ptr);
  }

  return result;
}

/**
 * @brief Thread-safe Python wrapper for the Polynomial background model evaluation.
 * 
 * @details Extends the zero-copy buffer protocol to ingest both the independent 
 * variable array and the arbitrary-length coefficient array dynamically passed 
 * from the Python sampling engine, strictly adhering to the re-entrant GIL constraints.
 */
py::array_t<double> py_evaluate_polynomial_background(py::array_t<double, py::array::c_style> f, py::array_t<double, py::array::c_style> coeffs) {
  py::buffer_info buf_f = f.request();
  py::buffer_info buf_c = coeffs.request();

  if (buf_f.ndim != 1 || buf_c.ndim != 1) {
    throw std::runtime_error("Both independent variable and coefficient arrays must be strictly 1-dimensional.");
  }
  
  if (buf_c.size == 0) {
    throw std::invalid_argument("Coefficient array dimension must be non-zero.");
  }

  auto result = py::array_t<double>(buf_f.size);
  py::buffer_info res_buf = result.request();

  const double* f_ptr = static_cast<const double*>(buf_f.ptr);
  const double* c_ptr = static_cast<const double*>(buf_c.ptr);
  double* res_ptr = static_cast<double*>(res_buf.ptr);

  std::size_t degree = buf_c.size - 1;

  {
    pybind11::gil_scoped_release release;
    evaluate_polynomial_background(f_ptr, buf_f.size, c_ptr, degree, res_ptr);
  }

  return result;
}

PYBIND11_MODULE(_core_models, m) {
  m.doc() = "C++ Python binding layer for computationally intensive forward models via pybind11.";

  // Standard bindings without the call_guard at the interface level to avoid 
  // asynchronous manipulation of Python objects without the GIL.
  m.def("evaluate_gaussian", &py_evaluate_gaussian, 
        "Evaluate Gaussian transient signal utilizing zero-copy memory mapping.",
        py::arg("f"), py::arg("amplitude"), py::arg("mu"), py::arg("sigma"));
        
  m.def("evaluate_pseudo_voigt", &py_evaluate_pseudo_voigt, 
        "Evaluate Pseudo-Voigt transient signal utilizing zero-copy memory mapping.",
        py::arg("f"), py::arg("amplitude"), py::arg("mu"), py::arg("sigma"), py::arg("gamma"));
        
  m.def("evaluate_exponential_background", &py_evaluate_exponential_background, 
        "Evaluate exponential decay background utilizing zero-copy memory mapping.",
        py::arg("f"), py::arg("amplitude"), py::arg("decay_constant"), py::arg("offset"));
        
  m.def("evaluate_polynomial_background", &py_evaluate_polynomial_background, 
        "Evaluate polynomial background utilizing zero-copy memory mapping.",
        py::arg("f"), py::arg("coeffs"));
}

} // namespace core
} // namespace bwsr_inference