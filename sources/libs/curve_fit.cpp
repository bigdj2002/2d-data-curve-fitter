#include "curve_fit.h"
#include <cmath>
#include <limits>
#include <cassert>
#include <cstring>

/**
 * @brief Solve the equation L * L^T * x = b using Cholesky decomposition
 *
 * This function solves the linear system where L is an n x n lower triangular matrix,
 * and x and b are n-size vectors. The solution is stored in the x array.
 *
 * @tparam T The data type (e.g., float, double)
 * @param n The size of the matrix/vector
 * @param l The lower triangular matrix L
 * @param x The vector x, initially used to store intermediate values y and final solution x
 * @param b The vector b
 */
template <typename T>
void solve_axb_cholesky(int n, const T *l, T *x, const T *b)
{
  // Solve L*y = b for y (where x[] is used to store y)
  // Forward substitution phase
  for (int i = 0; i < n; ++i)
  {
    T sum = 0;

    for (int j = 0; j < i; ++j)
    {
      sum += l[i * n + j] * x[j];
    }

    x[i] = (b[i] - sum) / l[i * n + i];
  }

  // Solve L^T*x = y for x (where x[] is used to store both y and x)
  // Backward substitution phase
  for (int i = n - 1; i >= 0; --i)
  {
    T sum = 0;

    for (int j = i + 1; j < n; ++j)
    {
      sum += l[j * n + i] * x[j];
    }

    x[i] = (x[i] - sum) / l[i * n + i];
  }
}

/**
 * @brief Perform Cholesky decomposition of a symmetric positive-definite matrix
 *
 * The Cholesky–Banachiewicz algorithm decomposes a symmetric positive-definite matrix A
 * into the product of a lower triangular matrix L and its transpose L^T.
 *
 * @tparam T The data type (e.g., float, double)
 * @param n The size of the matrix
 * @param l The lower triangular matrix L, which will store the decomposition result
 * @param a The input matrix A
 * @return true if the decomposition fails due to non-positive definiteness, false otherwise
 */
template <typename T>
bool cholesky_decomposition(int n, T *l, const T *a)
{
  int i, j, k;
  T sum;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < i; ++j)
    {
      sum = 0;

      for (k = 0; k < j; ++k)
      {
        sum += l[i * n + k] * l[j * n + k];
      }

      l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
    }

    sum = 0;

    for (k = 0; k < i; ++k)
    {
      sum += l[i * n + k] * l[i * n + k];
    }

    sum = a[i * n + i] - sum;

    if (sum < std::numeric_limits<T>::epsilon())
    {
      // Return true to indicate failure due to non-positive definiteness
      return true;
    }

    l[i * n + i] = std::sqrt(sum);
  }

  // Return false to indicate successful Cholesky decomposition
  return false;
}

/**
 * @brief Calculate the cost function for the Levenberg-Marquardt algorithm
 *
 * This function computes the sum of squared residuals between the model function values
 * and the actual data values.
 *
 * @tparam T The data type (e.g., float, double)
 * @param par The model parameters
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @param ndata The number of data points
 * @param model_func The model function
 * @return The cost value
 */
template <typename T>
T lm_calc_cost(
    const T *par,
    const T *ax,
    const T *ay,
    int ndata,
    T (*model_func)(T x, const T *params))
{
  T sum = 0;

  for (int i = 0; i < ndata; ++i)
  {
    T resi = model_func(ax[i], par) - ay[i];
    sum += (resi * resi);
  }
  return sum;
}

/**
 * @brief Levenberg-Marquardt algorithm for non-linear least squares optimization
 *
 * This function performs parameter optimization using the Levenberg-Marquardt algorithm,
 * which is a combination of gradient descent and the Gauss-Newton method.
 *
 * @tparam T The data type (e.g., float, double)
 * @param npar The number of parameters to optimize
 * @param params The array of initial parameter values, which will be updated with optimized values
 * @param ndat The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @param model_func The model function
 * @param grad_func The gradient function of the model
 * @param config The configuration settings for the algorithm
 * @param stat_out Optional pointer to a structure for storing statistics about the optimization process
 * @return true if the maximum number of iterations is reached, false otherwise
 */
template <typename T>
bool levenberg_marquardt_algorithm(
    int npar,
    T *params,
    int ndat,
    const T *ax,
    const T *ay,
    T (*model_func)(T x, const T *params),
    void (*grad_func)(T *grad_out, T x, const T *params),
    const curve_fit_lm_configs &config,
    curve_fit_lm_stats *stat_out)
{
  constexpr int MAX_PARAMS = 8;
  assert(npar <= MAX_PARAMS);

  const T up_factor = (T)config.up_factor;
  const T down_factor = (T)(1.0 / config.down_factor);
  const bool stat_enabled = stat_out != nullptr;

  T mat_jtj[MAX_PARAMS * MAX_PARAMS];  // transposed jacobian * jacobian
  T mat_jtr[MAX_PARAMS];               // transposed jacobian * ( y - y');   y' = model_func(ax, params)
  T mat_ltri[MAX_PARAMS * MAX_PARAMS]; // lower triangular matrix
  T lambda = config.init_lambda;
  T best_cost;
  int iter;

  if (stat_enabled)
  {
    stat_out->params_trail.clear();
    stat_out->params_trail.reserve(npar * (config.max_it + 1));
    stat_out->error_trail.clear();
    stat_out->error_trail.reserve(config.max_it + 1);
  }

  best_cost = lm_calc_cost(params, ax, ay, ndat, model_func);

  if (stat_enabled)
  {
    for (int i = 0; i < npar; ++i)
    {
      stat_out->params_trail.push_back((double)params[i]);
    }
    stat_out->error_trail.push_back((double)best_cost);
  }

  for (iter = 0; iter < config.max_it; ++iter)
  {
    // Calculate jtj and jtr
    std::memset(mat_jtj, 0, sizeof(T) * npar * npar);
    std::memset(mat_jtr, 0, sizeof(T) * npar);

    for (int ix = 0; ix < ndat; ++ix)
    {
      T gradients[MAX_PARAMS];

      grad_func(gradients, ax[ix], params);

      for (int i = 0; i < npar; ++i)
      {
        mat_jtr[i] += (ay[ix] - model_func(ax[ix], params)) * gradients[i];

        for (int j = 0; j <= i; ++j)
        {
          mat_jtj[i * npar + j] += gradients[i] * gradients[j];
        }
      }
    }

    // Update parameters with Levenberg Marquardt(LM) method.
    T new_params[MAX_PARAMS];
    T mult = 1 + lambda;
    bool ill_cond = true;
    T delta_cost;

    while (ill_cond &&
           (iter < config.max_it))
    {
      // reference:  A modified Marquardt subroutine for non-linear least squares simplified
      for (int i = 0; i < npar; ++i)
      {
        mat_jtj[i * npar + i] *= mult;
      }

      ill_cond = cholesky_decomposition(npar, mat_ltri, mat_jtj);

      if (!ill_cond)
      {
        T delta_params[MAX_PARAMS];

        solve_axb_cholesky(npar, mat_ltri, delta_params, mat_jtr);

        for (int i = 0; i < npar; ++i)
        {
          new_params[i] = params[i] + delta_params[i];
        }

        T cost = lm_calc_cost(new_params, ax, ay, ndat, model_func);
        if (stat_enabled)
        {
          for (int zz = 0; zz < npar; ++zz)
          {
            stat_out->params_trail.push_back((double)new_params[zz]);
          }
          stat_out->error_trail.push_back((double)cost);
        }

        if (cost < best_cost)
        {
          std::memcpy(params, new_params, sizeof(T) * npar);
          delta_cost = std::abs(best_cost - cost);
          best_cost = cost;
        }
        else
        {
          ill_cond = true;
        }
      }

      if (ill_cond)
      {
        mult = (1 + lambda * up_factor) / (1 + lambda);
        lambda *= up_factor;
        ++iter;
      }
    }

    lambda *= down_factor;

    if (!ill_cond &&
        (delta_cost < config.target_derr))
    {
      break;
    }
  }

  if (stat_enabled)
  {
    stat_out->num_iterations = iter;
  }

  return (iter == config.max_it);
}

/**
 * @brief Fit data to a hyperbolic model
 *
 * This function fits data to a hyperbolic model using the method of least squares.
 *
 * @tparam T The data type (e.g., float, double)
 * @param params The array to store the fitted parameters
 * @param n The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @return true if the fitting is ill-conditioned, false otherwise
 */
template <typename T>
bool hyperbolic_fit(
    T *params,
    int n,
    const T *ax,
    const T *ay)
{
  const T EPS = std::numeric_limits<T>::epsilon();
  bool ill_cond = false;

  T sum_lnx = 0;
  T sum_lny = 0;
  T sum_lnxlny = 0;
  T sum_lnx2 = 0;

  for (int i = 0; i < n; ++i)
  {
    bool zero_x = (ax[i] == 0);
    bool zero_y = (ay[i] == 0);
    ill_cond &= (zero_x || zero_y);

    T x = zero_x ? 0.00001 : ax[i];
    T y = zero_y ? 0.01 : ay[i];
    T lnx = std::log(x);
    T lny = std::log(y);

    sum_lnx += lnx;
    sum_lny += lny;
    sum_lnxlny += (lnx * lny);
    sum_lnx2 += (lnx * lnx);
  }

  T numer = sum_lny * sum_lnx - n * sum_lnxlny;
  T denom = n * sum_lnx2 - (sum_lnx * sum_lnx);

  ill_cond &= denom == 0;
  params[1] = numer / ((denom == 0) ? EPS : denom);
  params[0] = std::exp((sum_lny + params[1] * sum_lnx) / n);

  return ill_cond;
}

/**
 * @brief Fit data to a linear model
 *
 * This function fits data to a linear model using the method of least squares.
 *
 * @tparam T The data type (e.g., float, double)
 * @param par The array to store the fitted parameters
 * @param n The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @return true if the fitting is ill-conditioned, false otherwise
 */
template <typename T>
bool line_fit(
    T *par,
    int n,
    const T *ax,
    const T *ay)
{
  const T EPS = std::numeric_limits<T>::epsilon();
  bool ill_cond = false;

  T sum_x = 0;
  T sum_y = 0;
  T sum_xy = 0;
  T sum_x2 = 0;
  T sum_y2 = 0;

  for (int i = 0; i < n; ++i)
  {
    sum_x += ax[i];
    sum_y += ay[i];
    sum_xy += (ax[i] * ay[i]);
    sum_x2 += (ax[i] * ax[i]);
    sum_y2 += (ay[i] * ay[i]);
  }

  T numer = n * sum_xy - sum_x * sum_y;   // fixed
  T denom = n * sum_x2 - (sum_x * sum_x); // fixed

  ill_cond &= denom == 0;
  par[0] = numer / ((denom == 0) ? EPS : denom);
  par[1] = (sum_y - par[0] * sum_x) / n;

  return ill_cond;
}

/**
 * @brief Wrapper function for double-precision Levenberg-Marquardt curve fitting
 *
 * This function provides a double-precision interface to the Levenberg-Marquardt algorithm.
 *
 * @param npar The number of parameters to optimize
 * @param params The array of initial parameter values, which will be updated with optimized values
 * @param ndat The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @param model_func The model function
 * @param grad_func The gradient function of the model
 * @param config The configuration settings for the algorithm
 * @param stat_out Optional pointer to a structure for storing statistics about the optimization process
 * @return true if the maximum number of iterations is reached, false otherwise
 */
bool curve_fit_lm_d(
    int npar,
    double *params,
    int ndat,
    const double *ax,
    const double *ay,
    LmFunc_d model_func,
    LmGrad_d grad_func,
    const curve_fit_lm_configs &config,
    curve_fit_lm_stats *stat_out)
{
  return levenberg_marquardt_algorithm<double>(
      npar,
      params,
      ndat,
      ax,
      ay,
      model_func,
      grad_func,
      config,
      stat_out);
}

/**
 * @brief Wrapper function for single-precision Levenberg-Marquardt curve fitting
 *
 * This function provides a single-precision interface to the Levenberg-Marquardt algorithm.
 *
 * @param npar The number of parameters to optimize
 * @param params The array of initial parameter values, which will be updated with optimized values
 * @param ndat The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @param model_func The model function
 * @param grad_func The gradient function of the model
 * @param config The configuration settings for the algorithm
 * @param stat_out Optional pointer to a structure for storing statistics about the optimization process
 * @return true if the maximum number of iterations is reached, false otherwise
 */
bool curve_fit_lm_f(
    int npar,
    float *params,
    int ndat,
    const float *ax,
    const float *ay,
    LmFunc_f model_func,
    LmGrad_f grad_func,
    const curve_fit_lm_configs &config,
    curve_fit_lm_stats *stat_out)
{
  return levenberg_marquardt_algorithm<float>(
      npar,
      params,
      ndat,
      ax,
      ay,
      model_func,
      grad_func,
      config,
      stat_out);
}

/**
 * @brief Wrapper function for double-precision hyperbolic model fitting
 *
 * This function provides a double-precision interface to the hyperbolic model fitting function.
 *
 * @param params The array to store the fitted parameters
 * @param num_data The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @return true if the fitting is ill-conditioned, false otherwise
 */
bool hyperbolic_fit_d(
    double *params,
    int num_data,
    const double *ax,
    const double *ay)
{
  return hyperbolic_fit<double>(params, num_data, ax, ay);
}

/**
 * @brief Wrapper function for single-precision hyperbolic model fitting
 *
 * This function provides a single-precision interface to the hyperbolic model fitting function.
 *
 * @param params The array to store the fitted parameters
 * @param num_data The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @return true if the fitting is ill-conditioned, false otherwise
 */
bool hyperbolic_fit_f(
    float *params,
    int num_data,
    const float *ax,
    const float *ay)
{
  return hyperbolic_fit<float>(params, num_data, ax, ay);
}

/**
 * @brief Wrapper function for double-precision linear model fitting
 *
 * This function provides a double-precision interface to the linear model fitting function.
 *
 * @param params The array to store the fitted parameters
 * @param num_data The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @return true if the fitting is ill-conditioned, false otherwise
 */
bool line_fit_d(
    double *params,
    int num_data,
    const double *ax,
    const double *ay)
{
  return line_fit(params, num_data, ax, ay);
}

/**
 * @brief Wrapper function for single-precision linear model fitting
 *
 * This function provides a single-precision interface to the linear model fitting function.
 *
 * @param params The array to store the fitted parameters
 * @param num_data The number of data points
 * @param ax The x-coordinates of the data points
 * @param ay The y-coordinates of the data points
 * @return true if the fitting is ill-conditioned, false otherwise
 */
bool line_fit_f(
    float *params,
    int num_data,
    const float *ax,
    const float *ay)
{
  return line_fit(params, num_data, ax, ay);
}
