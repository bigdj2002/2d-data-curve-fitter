#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace utility
{
    /**
     * @brief Clips a value between a minimum and a maximum value.
     * 
     * @tparam T     Type of the value.
     * @param val    The value to clip.
     * @param minV   The minimum value.
     * @param maxV   The maximum value.
     * @return       Clipped value.
     */
    template <typename T>
    T clip3(const T val, const T minV, const T maxV)
    {
        return std::max(std::min(maxV, val), minV);
    }

    /**
     * @brief Calculates the mean squared error (MSE) between two arrays.
     * 
     * @tparam T     Type of the array elements.
     * @param a1     First array.
     * @param a2     Second array.
     * @param n      Number of elements in the arrays.
     * @return       Mean squared error.
     */
    template <typename T>
    T calc_mse(const T *a1, const T *a2, int n)
    {
        T sum = 0;
        for (int i = 0; i < n; ++i)
        {
            T diff = a1[i] - a2[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    /**
     * @brief Calculates the mean of an array.
     * 
     * @tparam T     Type of the array elements.
     * @param v      Array.
     * @param n      Number of elements in the array.
     * @return       Mean of the array.
     */
    template <typename T>
    T calc_mean(const T *v, int n)
    {
        T sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += v[i];
        }
        return sum / n;
    }

    /**
     * @brief Calculates the standard deviation of an array.
     * 
     * @tparam T     Type of the array elements.
     * @param v      Array.
     * @param n      Number of elements in the array.
     * @return       Standard deviation of the array.
     */
    template <typename T>
    T calc_std(const T *v, int n)
    {
        if (n <= 1)
        {
            return T(0);
        }

        T mean = calc_mean(v, n);
        T var = (T)0;

        for (int i = 0; i < n; ++i)
        {
            T diff = (v[i] - mean);
            var += (diff * diff);
        }

        var = var / (n - 1);
        return std::sqrt(var);
    }

    /**
     * @brief Calculates the covariance between two arrays.
     * 
     * @tparam T     Type of the array elements.
     * @param ax     First array.
     * @param ay     Second array.
     * @param n      Number of elements in the arrays.
     * @return       Covariance between the two arrays.
     */
    template <typename T>
    T calc_cov(const T *ax, const T *ay, int n)
    {
        T mean_x = calc_mean(ax, n);
        T mean_y = calc_mean(ay, n);

        T sxy = 0;
        for (int i = 0; i < n; ++i)
        {
            sxy += (ax[i] - mean_x) * (ay[i] - mean_y);
        }

        return sxy / n;
    }

    /**
     * @brief Calculates the Pearson correlation coefficient (PCC) between two arrays.
     * 
     * @tparam T     Type of the array elements.
     * @param ax     First array.
     * @param ay     Second array.
     * @param n      Number of elements in the arrays.
     * @return       Pearson correlation coefficient.
     */
    template <typename T>
    T calc_pcc(const T *ax, const T *ay, int n)
    {
        T cov_xy = calc_cov(ax, ay, n);
        T std_x = calc_std(ax, n);
        T std_y = calc_std(ay, n);

        if (std_x == 0)
        {
            std_x = std::numeric_limits<T>::epsilon();
        }

        if (std_y == 0)
        {
            std_y = std::numeric_limits<T>::epsilon();
        }

        T pcc = cov_xy / (std_x * std_y);

        return clip3(pcc, T(-1.0), T(1.0));
    }

    /**
     * @brief Calculates the R-squared value between original and predicted arrays.
     * 
     * @param aorg   Original array.
     * @param apred  Predicted array.
     * @param n      Number of elements in the arrays.
     * @return       R-squared value.
     */
    double calc_rsquared(const double *aorg, const double *apred, int n)
    {
        double mean_org = calc_mean(aorg, n);
        double ssTotal = 0;
        double ssResidual = 0;

        for (int i = 0; i < n; ++i)
        {
            ssTotal += ((aorg[i] - mean_org) * (aorg[i] - mean_org));
            ssResidual += ((aorg[i] - apred[i]) * (aorg[i] - apred[i]));
        }

        if (ssTotal == 0)
        {
            ssTotal = std::numeric_limits<double>::epsilon();
        }

        double rSquared = 1.0 - (ssResidual / ssTotal);
        return rSquared;
    }

    /**
     * @brief Class for computing Exponential Moving Average (EMA).
     */
    class ExponentialMovingAverage
    {
    private:
        double alpha;        ///< Decay factor
        double ema;          ///< Current EMA value
        bool is_initialized; ///< Whether the EMA is initialized or not

    public:
        /**
         * @brief Constructor for ExponentialMovingAverage.
         * 
         * @param alpha    Decay factor.
         */
        ExponentialMovingAverage(double alpha) : alpha(alpha), ema(0.0), is_initialized(false) {}

        /**
         * @brief Updates the EMA with a new value.
         * 
         * @param newValue  New value to include in the EMA.
         */
        void update(double newValue)
        {
            if (!is_initialized)
            {
                ema = newValue;
                is_initialized = true;
            }
            else
            {
                ema = alpha * newValue + (1 - alpha) * ema;
            }
        }

        /**
         * @brief Gets the current EMA value.
         * 
         * @return Current EMA value.
         */
        double get_value() const
        {
            return ema;
        }
    };
}
