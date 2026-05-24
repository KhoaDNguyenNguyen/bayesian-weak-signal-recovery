#include <cmath>
#include <stdexcept>
#include <cstddef>

namespace bwsr_inference {
namespace core {

/**
 * @brief Computes the power spectral density of a Gaussian transient signal.
 */
void evaluate_gaussian(const double* f, std::size_t size, double amplitude, double mu, double sigma, double* output) {
    if (sigma <= 0.0) {
        throw std::invalid_argument("Physical constraint violation: Gaussian width must be strictly positive.");
    }
    const double variance_inv = 1.0 / (2.0 * sigma * sigma);
    for (std::size_t i = 0; i < size; ++i) {
        const double diff = f[i] - mu;
        output[i] = amplitude * std::exp(-diff * diff * variance_inv);
    }
}

/**
 * @brief Computes the pseudo-Voigt profile for resonance line shapes.
 */
void evaluate_pseudo_voigt(const double* f, std::size_t size, double amplitude, double mu, double sigma, double gamma, double* output) {
    if (sigma <= 0.0 && gamma <= 0.0) {
        throw std::invalid_argument("Physical constraint violation: Broadening parameters must be strictly positive.");
    }
    const double w_g = sigma * std::sqrt(8.0 * std::log(2.0));
    const double w_l = 2.0 * gamma;
    const double w_v = 0.5346 * w_l + std::sqrt(0.2166 * w_l * w_l + w_g * w_g);
    
    if (w_v == 0.0) {
        throw std::invalid_argument("Numerical instability: Derived Voigt width evaluates to zero.");
    }

    const double ratio = w_l / w_v;
    const double eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio;
    const double gaussian_norm = 4.0 * std::log(2.0) / (w_v * w_v);
    const double lorentzian_norm = w_v * w_v;

    for (std::size_t i = 0; i < size; ++i) {
        const double diff = f[i] - mu;
        const double diff_sq = diff * diff;
        const double g_val = std::exp(-gaussian_norm * diff_sq);
        const double l_val = lorentzian_norm / (lorentzian_norm + 4.0 * diff_sq);
        output[i] = amplitude * ((1.0 - eta) * g_val + eta * l_val);
    }
}

/**
 * @brief Computes an exponentially decaying stochastic noise background.
 */
void evaluate_exponential_background(const double* f, std::size_t size, double amplitude, double decay_constant, double offset, double* output) {
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = amplitude * std::exp(-decay_constant * f[i]) + offset;
    }
}

/**
 * @brief Computes a polynomial stochastic noise background utilizing Horner's method.
 */
void evaluate_polynomial_background(const double* f, std::size_t size, const double* coeffs, std::size_t degree, double* output) {
    for (std::size_t i = 0; i < size; ++i) {
        double result = coeffs[degree];
        for (std::ptrdiff_t j = degree - 1; j >= 0; --j) {
            result = result * f[i] + coeffs[j];
        }
        output[i] = result;
    }
}

} // namespace core
} // namespace bwsr_inference