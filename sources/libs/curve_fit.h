#pragma once
#include <vector>

/**
 * @brief Structure to hold statistics of the Levenberg-Marquardt fitting process
 */
struct curve_fit_lm_stats
{
    int num_iterations;                  ///< Number of iterations performed
    std::vector<double> params_trail;    ///< Trail of parameter values during optimization
    std::vector<double> error_trail;     ///< Trail of error values during optimization
};

/**
 * @brief Structure to hold configuration settings for the Levenberg-Marquardt algorithm
 */
struct curve_fit_lm_configs
{
    int max_it = 10000;                  ///< Maximum number of iterations
    double init_lambda = 0.0001;         ///< Initial lambda value for the damping factor
    double up_factor = 10;               ///< Factor to increase the damping parameter
    double down_factor = 10;             ///< Factor to decrease the damping parameter
    double target_derr = 1e-30;          ///< Target change in error for convergence
};

// Type definitions for function pointers for the model and gradient functions
typedef double (*LmFunc_d)(double x, const double *params);
typedef void (*LmGrad_d)(double *grad_out, double x, const double *params);
typedef float (*LmFunc_f)(float x, const float *params);
typedef void (*LmGrad_f)(float *grad_out, float x, const float *params);

/**
 * @brief Performs curve fitting using the Levenberg-Marquardt algorithm (double precision)
 * 
 * @param num_params    Number of parameters in the model function.
 * @param params        Array of initial parameter estimates for the model function.
 * @param num_data      Number of data points.
 * @param ax            Array of x-coordinates of the data points.
 * @param ay            Array of y-coordinates of the data points.
 * @param model_func    Pointer to the user-defined function that computes model values.
 * @param grad_func     Pointer to the user-defined function that computes the gradient of the model function.
 * @param config        Configuration settings for the Levenberg-Marquardt algorithm.
 * @param stat_out      Optional pointer to a structure for storing statistics about the fitting process.
 * 
 * @return              True if the maximum number of iterations is reached, false otherwise.
 */
extern bool curve_fit_lm_d(
    int num_params,
    double *params,
    int num_data,
    const double *ax,
    const double *ay,
    LmFunc_d model_func,
    LmGrad_d grad_func,
    const curve_fit_lm_configs &config,
    curve_fit_lm_stats* stat_out = nullptr);

/**
 * @brief Performs curve fitting using the Levenberg-Marquardt algorithm (single precision)
 * 
 * @param num_params    Number of parameters in the model function.
 * @param params        Array of initial parameter estimates for the model function.
 * @param num_data      Number of data points.
 * @param ax            Array of x-coordinates of the data points.
 * @param ay            Array of y-coordinates of the data points.
 * @param model_func    Pointer to the user-defined function that computes model values.
 * @param grad_func     Pointer to the user-defined function that computes the gradient of the model function.
 * @param config        Configuration settings for the Levenberg-Marquardt algorithm.
 * @param stat_out      Optional pointer to a structure for storing statistics about the fitting process.
 * 
 * @return              True if the maximum number of iterations is reached, false otherwise.
 */
extern bool curve_fit_lm_f(
    int num_params,
    float *params,
    int num_data,
    const float *ax,
    const float *ay,
    LmFunc_f model_func,
    LmGrad_f grad_func,
    const curve_fit_lm_configs &config,
    curve_fit_lm_stats* stat_out = nullptr);

/**
 * @brief Performs Hyperbolic fitting (double precision)
 * 
 * This function fits data to a hyperbolic model using the method of least squares.
 * The target function is y = a * pow(x, -b).
 * 
 * @param params        Array to store the fitted parameters.
 * @param num_data      Number of data points.
 * @param ax            Array of x-coordinates of the data points.
 * @param ay            Array of y-coordinates of the data points.
 * 
 * @return              True if the fitting is ill-conditioned, false otherwise.
 */
extern bool hyperbolic_fit_d(
    double *params,
    int num_data,
    const double *ax,
    const double *ay);

/**
 * @brief Performs Hyperbolic fitting (single precision)
 * 
 * This function fits data to a hyperbolic model using the method of least squares.
 * The target function is y = a * pow(x, -b).
 * 
 * @param params        Array to store the fitted parameters.
 * @param num_data      Number of data points.
 * @param ax            Array of x-coordinates of the data points.
 * @param ay            Array of y-coordinates of the data points.
 * 
 * @return              True if the fitting is ill-conditioned, false otherwise.
 */
extern bool hyperbolic_fit_f(
    float *params,
    int num_data,
    const float *ax,
    const float *ay);

/**
 * @brief Performs Linear fitting (double precision)
 * 
 * This function fits data to a linear model using the method of least squares.
 * The target function is y = a * x + b.
 * 
 * @param params        Array to store the fitted parameters.
 * @param num_data      Number of data points.
 * @param ax            Array of x-coordinates of the data points.
 * @param ay            Array of y-coordinates of the data points.
 * 
 * @return              True if the fitting is ill-conditioned, false otherwise.
 */
extern bool line_fit_d(
    double *params,
    int num_data,
    const double *ax,
    const double *ay);

/**
 * @brief Performs Linear fitting (single precision)
 * 
 * This function fits data to a linear model using the method of least squares.
 * The target function is y = a * x + b.
 * 
 * @param params        Array to store the fitted parameters.
 * @param num_data      Number of data points.
 * @param ax            Array of x-coordinates of the data points.
 * @param ay            Array of y-coordinates of the data points.
 * 
 * @return              True if the fitting is ill-conditioned, false otherwise.
 */
extern bool line_fit_f(
    float *params,
    int num_data,
    const float *ax,
    const float *ay);
