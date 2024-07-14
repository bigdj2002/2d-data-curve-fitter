#include "data_set.h"
#include "models/models.h"
#include "libs/curve_fit.h"

#include <iomanip>
#include <algorithm>

static void check_qp_validation(DataSet &ds,
                                std::vector<double> &err,
                                std::vector<double> &enc_qp,
                                std::vector<double> &lnLamda,
                                double *params)
{
    LineModel line_model;
    LmFunc_d line_model_fit = line_model.model;
    LmGrad_d line_model_grad = line_model.gradient;

    std::vector<double> lambda_filtered;
    std::vector<double> qp_filtered;

    /**
     * \brief: Test moving average to filter the unnecessary data
    */
    std::vector<double> pred_full_qp;
    pred_full_qp.reserve(ds.getDim0() * ds.getDim1());

    ds.write_json_data("/mnt/sDisk/tests/test_lm/test_lm/out/lnLambda-encQP.json", lnLamda.data(), enc_qp.data(), ds.getDim0() * ds.getDim1());

    line_fit_d(params,
               ds.getDim0() * ds.getDim1(),
               lnLamda.data(),
               enc_qp.data());

    // params[0] = 4.2005;
    // params[1] = 13.7122;
    // params[0] = 2.2849801107181795;
    // params[1] = 18.62163698179026;

    for (int i = 0; i < ds.getDim0(); ++i)
    {
        double diff = 0;
        const double *qp_ptr = ds.getConstEncQp().data() + i * ds.getDim1();
        const double *lambda_ptr = ds.getConstLnLambda().data() + i * ds.getDim1();
        for (int j = 0; j < ds.getDim1(); ++j)
        {
            double pred_qp = std::floor(params[0] * lambda_ptr[j] + params[1]);
            pred_full_qp.push_back(pred_qp);
            diff += std::fabs(pred_qp - qp_ptr[j]);
        }

        diff = diff / ds.getDim1();
        err.push_back(diff);
    }

    ds.write_json_data("/mnt/sDisk/tests/test_lm/test_lm/out/lnLambda-predQP.json", lnLamda.data(), pred_full_qp.data(), ds.getDim0() * ds.getDim1());
}

class Tester
{
public:
    void performTest(DataSet &ds, int mode, std::string title);

private:
    void modeH2Closed(DataSet &ds)
    {
        Hy2Model hy2_model;
        for (int i = 0; i < ds.getDim0(); ++i)
        {
            const double *org_fax = ds.getAxFiltered().data() + i * ds.getDim1Filtered();
            const double *org_fay = ds.getAyFiltered().data() + i * ds.getDim1Filtered();

            hyperbolic_fit_d(params,
                             ds.getDim1Filtered(),
                             org_fax,
                             org_fay);

            const double *org_ax = ds.getAx().data() + i * ds.getDim1();
            const double *org_ay = ds.getAy().data() + i * ds.getDim1();

            for (int i = 0; i < valset_count; ++i)
            {
                if (ds.getAyName() == "vmaf")
                    pred_ay[i] = utility::clip3(hy2_model.model(org_ax[i], params), 0.0, 1.0);
                else
                    pred_ay[i] = hy2_model.model(org_ax[i], params);
            }
            if (ds.getAyName() == "vmaf")
                err = utility::calc_mse(org_ay, pred_ay, valset_count) * 10000.0;
            else
                err = utility::calc_mse(org_ay, pred_ay, valset_count);
            aerr.push_back(err);

            for (int i = 0; i < ds.getDim1(); i++)
            {
                double lnLambda = std::log(params[0] * params[1] * std::pow(org_ax[i], -params[1] - 1));
                ds.getLnLambda().push_back(lnLambda);
            }
        }

        check_qp_validation(ds, merr, ds.getEncQp(), ds.getLnLambda(), qp_params);

        reportResults(aerr, merr, params, qp_params);
        aerr.clear();
        merr.clear();
    }

    void modeH2Iter(DataSet &ds)
    {
    }

    void modeH3Iter(DataSet &dysize)
    {
    }

    void modeH2ClosedH2Iter(DataSet &ds)
    {
        Hy2Model model;
        LmFunc_d model_func = model.model;
        LmGrad_d grad_func = model.gradient;
        for (int i = 0; i < ds.getDim0(); ++i)
        {
            const double *org_fax = ds.getAxFiltered().data() + i * ds.getDim1Filtered();
            const double *org_fay = ds.getAyFiltered().data() + i * ds.getDim1Filtered();
            hyperbolic_fit_d(params,
                             ds.getDim1Filtered(),
                             org_fax,
                             org_fay);

            lm_config.init_lambda = 0.0001;
            lm_config.target_derr = 0.00001;
            lm_config.up_factor = 8;
            lm_config.down_factor = 8;
            lm_config.max_it = 40;
            curve_fit_lm_d(
                2,
                params,
                ds.getDim1Filtered(),
                org_fax,
                org_fay,
                model_func,
                grad_func,
                lm_config);

            const double *org_ax = ds.getAx().data() + i * ds.getDim1();
            const double *org_ay = ds.getAy().data() + i * ds.getDim1();

            for (int i = 0; i < valset_count; ++i)
            {
                if (ds.getAyName() == "vmaf")
                    pred_ay[i] = utility::clip3(model.model(org_ax[i], params), 0.0, 1.0);
                else
                    pred_ay[i] = model.model(org_ax[i], params);
            }
            if (ds.getAyName() == "vmaf")
                err = utility::calc_mse(org_ay, pred_ay, valset_count) * 10000.0;
            else
                err = utility::calc_mse(org_ay, pred_ay, valset_count);
            aerr.push_back(err);

            for (int i = 0; i < ds.getDim1(); i++)
            {
                double lnLambda = std::log(params[0] * params[1] * std::pow(org_ax[i], -params[1] - 1));
                ds.getLnLambda().push_back(lnLambda);
            }
        }

        check_qp_validation(ds, merr, ds.getEncQp(), ds.getLnLambda(), qp_params);

        reportResults(aerr, merr, params, qp_params);
        aerr.clear();
        merr.clear();
    }

    void modeH2ClosedH3Iter(DataSet &ds)
    {
        Hy3Model model;
        LmFunc_d model_func = model.model;
        LmGrad_d grad_func = model.gradient;
        for (int i = 0; i < ds.getDim0(); ++i)
        {
            const double *org_fax = ds.getAxFiltered().data() + i * ds.getDim1Filtered();
            const double *org_fay = ds.getAyFiltered().data() + i * ds.getDim1Filtered();
            hyperbolic_fit_d(params,
                             ds.getDim1Filtered(),
                             org_fax,
                             org_fay);

            lm_config.init_lambda = 0.0001;
            lm_config.target_derr = 0.00001;
            lm_config.up_factor = 8;
            lm_config.down_factor = 8;
            lm_config.max_it = 40;
            curve_fit_lm_d(
                3,
                params,
                ds.getDim1Filtered(),
                org_fax,
                org_fay,
                model_func,
                grad_func,
                lm_config);

            const double *org_ax = ds.getAx().data() + i * ds.getDim1();
            const double *org_ay = ds.getAy().data() + i * ds.getDim1();

            for (int i = 0; i < valset_count; ++i)
            {
                if (ds.getAyName() == "vmaf")
                    pred_ay[i] = utility::clip3(model.model(org_ax[i], params), 0.0, 1.0);
                else
                    pred_ay[i] = model.model(org_ax[i], params);
            }
            if (ds.getAyName() == "vmaf")
                err = utility::calc_mse(org_ay, pred_ay, valset_count) * 10000.0;
            else
                err = utility::calc_mse(org_ay, pred_ay, valset_count);
            aerr.push_back(err);

            for (int i = 0; i < ds.getDim1(); i++)
            {
                double lnLambda = std::log(params[0] * params[1] * std::pow(org_ax[i], -params[1] - 1));
                ds.getLnLambda().push_back(lnLambda);
            }
        }

        check_qp_validation(ds, merr, ds.getEncQp(), ds.getLnLambda(), qp_params);

        reportResults(aerr, merr, params, qp_params);
        aerr.clear();
        merr.clear();
    }

private:
    void reportResults(const std::vector<double> &aerr,
                       const std::vector<double> &merr,
                       double *params,
                       double *qp_params)
    {
        double mse_mean = utility::calc_mean(aerr.data(), aerr.size());
        double mse_std = utility::calc_std(aerr.data(), aerr.size());
        double mse_min = *std::min_element(aerr.begin(), aerr.end());
        double mse_max = *std::max_element(aerr.begin(), aerr.end());
        int max_where = (int)(std::max_element(aerr.begin(), aerr.end()) - aerr.begin());

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "* Params:" << params[0] << ", " << params[1] << ", " << params[2] << std::endl;
        std::cout << "* RMSE mean = " << std::sqrt(mse_mean) << "\n";
        std::cout << "* RMSE std = " << std::sqrt(mse_std) << "\n";
        std::cout << "* RMSE min = " << std::sqrt(mse_min) << "\n";
        std::cout << "* RMSE max = " << std::sqrt(mse_max) << ", at index = " << max_where << "\n";
        std::cout << std::endl;

        double diff_mean = utility::calc_mean(merr.data(), merr.size());
        double diff_std = utility::calc_std(merr.data(), merr.size());
        double diff_min = *std::min_element(merr.begin(), merr.end());
        double diff_max = *std::max_element(merr.begin(), merr.end());
        max_where = (int)(std::max_element(merr.begin(), merr.end()) - merr.begin());

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "* Params:" << qp_params[0] << ", " << qp_params[1] << ", " << qp_params[2] << std::endl;
        std::cout << "* QP diff mean = " << std::sqrt(diff_mean) << "\n";
        std::cout << "* QP diff std = " << std::sqrt(diff_std) << "\n";
        std::cout << "* QP diff min = " << std::sqrt(diff_min) << "\n";
        std::cout << "* QP diff max = " << std::sqrt(diff_max) << ", at index = " << max_where << "\n";
        std::cout << std::endl;
    }

    void init_config()
    {
        valset_count = 11; // 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41
        model_func = nullptr;
        grad_func = nullptr;
        npar = 0;
        memset(params, 0, sizeof(int) * 8);
        memset(qp_params, 0, sizeof(int) * 8);
        memset(pred_ay, 0, sizeof(double) * 32);
        err = 0;
    }

    void init_lm_config()
    {
        lm_config.init_lambda = 0.0001;
        lm_config.target_derr = 0.00001;
        lm_config.up_factor = 8;
        lm_config.down_factor = 8;
        lm_config.max_it = 60;
    }

    void init_metric(DataSet &ds)
    {
        aerr.reserve(ds.getDim0());
        merr.reserve(ds.getDim0() * 5);
    }

private:
    int valset_count;
    std::string title;
    LmFunc_d model_func;
    LmGrad_d grad_func;
    int npar;
    double params[8];
    double qp_params[8];
    double pred_ay[32];
    double err;

    std::vector<double> aerr; //
    std::vector<double> merr; //

    curve_fit_lm_configs lm_config = {};
};