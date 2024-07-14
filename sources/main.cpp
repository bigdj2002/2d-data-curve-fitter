#include "model_tester.h"
#include "config.h"

#include <numeric> // for std::iota

void Tester::performTest(DataSet &ds, int mode, std::string title)
{
    init_config();
    init_lm_config();
    init_metric(ds);

    std::cout << "***  " << title << "  ***" << std::endl;
    std::cout << "* X-axis label: " << ds.getAxName() << std::endl;
    std::cout << "* Y-axis label: " << ds.getAyName() << std::endl;

    switch (mode)
    {
    case 0:
        modeH2Closed(ds);
        break;
    case 1:
        modeH2Iter(ds);
        break;
    case 2:
        modeH3Iter(ds);
        break;
    case 3:
        modeH2ClosedH2Iter(ds);
        break;
    case 4:
        modeH2ClosedH3Iter(ds);
        break;
    }

    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    AppCfgs app_configs;
    if (!app_configs.parseCfg(argc, argv))
    {
        throw std::runtime_error{"Failed in parsing arguments"};
        return -1;
    }

    DataSet ds;
    std::vector<int> filter(30);                // Create a vector with 31 elements
    std::iota(filter.begin(), filter.end(), 0); // Fill with values from 0 to 30

    // std::vector<int> filter = {0, 3, 5, 8, 10};
    // std::vector<int> filter = {5, 10};

    std::string ax_name = "bpp";
    std::string ay_name = "mse";
    std::string feature_path = "data/encoding_data2.json";
    ds.parse_json_data(feature_path.c_str(), filter, ds, ax_name, ay_name);

    std::vector<int> ax = app_configs.parse_string2int(app_configs.get_x_axis());
    std::vector<int> ay = app_configs.parse_string2int(app_configs.get_y_axis());

    Tester tester;
    tester.performTest(ds, 0, "Hyperbolic 2-param closed model");
    tester.performTest(ds, 3, "Hyperbolic 2-param closed model → Hyperbolic 2-param iterative model (L-M model)");
    tester.performTest(ds, 4, "Hyperbolic 2-param closed model → Hyperbolic 3-param iterative model (L-M model)");

    return 0;
}