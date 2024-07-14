#pragma once
#include <cmath>
#include <algorithm>
#include "math_util.h"

/**
 * @brief Model class for a simple linear model y = a * x + b.
 */
class LineModel
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return p[0] * x + p[1];
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b].
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   */
  static void gradient(double *o, double x, const double *p)
  {
    o[0] = x;
    o[1] = 1.0;
  }
};

/**
 * @brief Model class for a hyperbolic model y = a * x^(-b).
 */
class Hy2Model
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return p[0] * std::pow(x, -p[1]);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b].
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   */
  static void gradient(double *o, double x, const double *p)
  {
    o[0] = std::pow(x, -p[1]);
    o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
  }
};

/**
 * @brief Model class for a capped hyperbolic model y = min(a * x^(-b), 1).
 */
class Hy2CapModel
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return std::min(p[0] * std::pow(x, -p[1]), 1.0);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b].
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   */
  static void gradient(double *o, double x, const double *p)
  {
    double criterion = (p[0] > 0 && p[1] > 0) ? std::pow(1.0 / p[0], -1.0 / p[1]) : 1.0;

    if (x > criterion)
    {
      o[0] = o[1] = 0.0;
    }
    else
    {
      o[0] = std::pow(x, -p[1]);
      o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
    }
  }
};

/**
 * @brief Model class for a hyperbolic model with an offset y = a * x^(-b) + c.
 */
class Hy3Model
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return p[0] * std::pow(x, -p[1]) + p[2];
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b, ∂y/∂c].
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   */
  static void gradient(double *o, double x, const double *p)
  {
    o[0] = std::pow(x, -p[1]);
    o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
    o[2] = 1.0;
  }
};

/**
 * @brief Model class for a capped hyperbolic model with an offset y = max(a * x^(-b) + c, 0).
 */
class Hy3CapModel
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return std::max(p[0] * std::pow(x, -p[1]) + p[2], 0.0);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b, ∂y/∂c].
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   */
  static void gradient(double *o, double x, const double *p)
  {
    double criterion = (p[0] > 0 && p[1] > 0) ? std::pow(-p[2] / p[0], -1.0 / p[1]) : 0;

    if (x < criterion)
    {
      o[0] = o[1] = o[2] = 0.0;
    }
    else
    {
      o[0] = std::pow(x, -p[1]);
      o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
      o[2] = 1.0;
    }
  }
};

/**
 * @brief Model class for a capped hyperbolic model with an offset y = max(a * x^(-b) + c, 0) on the left side.
 */
class Hy3CapLModel
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return std::max(p[0] * std::pow(x, -p[1]) + p[2], 0.0);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b, ∂y/∂c].
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   */
  static void gradient(double *o, double x, const double *p)
  {
    double criterion = (p[0] > 0 && p[1] > 0) ? std::pow(-p[2] / p[0], -1.0 / p[1]) : 0;
    if (x < criterion)
    {
      o[0] = o[1] = o[2] = 0.0;
    }
    else
    {
      o[0] = std::pow(x, -p[1]);
      o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
      o[2] = 1.0;
    }
  }
};

/**
 * @brief Model class for a capped hyperbolic model with an offset y = min(a * x^(-b) + c, 1) on the right side.
 */
class Hy3CapRModel
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return std::min(p[0] * std::pow(x, -p[1]) + p[2], 1.0);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b, ∂y/∂c].
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   */
  static void gradient(double *o, double x, const double *p)
  {
    double criterion = (p[0] > 0 && p[1] > 0) ? std::pow((1.0 - p[2]) / p[0], -1.0 / p[1]) : 1.0;
    if (x > criterion)
    {
      o[0] = o[1] = o[2] = 0.0;
    }
    else
    {
      o[0] = std::pow(x, -p[1]);
      o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
      o[2] = 1.0;
    }
  }
};

/**
 * @brief Model class for a fully capped hyperbolic model with an offset y = clip(a * x^(-b) + c, 0, 1).
 */
class Hy3CapLRModel
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return utility::clip3(p[0] * std::pow(x, -p[1]) + p[2], 0.0, 1.0);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b, ∂y/∂c].
   * @param x   Input value.
   * @param p   Array of parameters [a, b, c].
   */
  static void gradient(double *o, double x, const double *p)
  {
    double criterion_l = (p[0] > 0 && p[1] > 0) ? std::pow(-p[2] / p[0], -1.0 / p[1]) : 0;
    double criterion_r = (p[0] > 0 && p[1] > 0) ? std::pow((1.0 - p[2]) / p[0], -1.0 / p[1]) : 1.0;
    if (x < criterion_l || x > criterion_r)
    {
      o[0] = o[1] = o[2] = 0.0;
    }
    else
    {
      o[0] = std::pow(x, -p[1]);
      o[1] = -p[0] * p[1] * std::pow(x, -p[1] - 1);
      o[2] = 1.0;
    }
  }
};

/**
 * @brief Model class for a logarithmic model y = floor(a * log(x) + b).
 */
class lambda2Model
{
public:
  /**
   * @brief Computes the model value for a given x.
   *
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   * @return    Model value.
   */
  static double model(double x, const double *p)
  {
    return std::floor(p[0] * std::log(x) + p[1]);
  }

  /**
   * @brief Computes the gradient of the model with respect to the parameters.
   *
   * @param o   Output gradient array [∂y/∂a, ∂y/∂b].
   * @param x   Input value.
   * @param p   Array of parameters [a, b].
   */
  static void gradient(double *o, double x, const double *p)
  {
    o[0] = std::log(x);
    o[1] = 1;
  }
};
