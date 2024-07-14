#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <json/json.h>

namespace fs = std::filesystem;

class DataSet
{
public:
    int parse_json_data(const char *path,
                        const std::vector<int> filters,
                        DataSet &out,
                        std::string &ax_name,
                        std::string &ay_name);
    void write_json_data(const char *path,
                         const double *ax,
                         const double *ay,
                         int n);

public:
    int getDim0() const { return dim0; }
    int getDim1() const { return dim1; }
    int getDim1Filtered() const { return dim1_filtered; }
    const std::vector<double> &getAx() const { return ax; }
    const std::vector<double> &getAy() const { return ay; }
    const std::vector<double> &getAxFiltered() const { return ax_filtered; }
    const std::vector<double> &getAyFiltered() const { return ay_filtered; }
    const std::string &getAxName() const { return ax_name; }
    const std::string &getAyName() const { return ay_name; }
    const std::vector<double> &getConstEncQp() const { return encoding_qp; }
    const std::vector<double> &getConstLnLambda() const { return lnLambda; }
    std::vector<double> &getEncQp() { return encoding_qp; }
    std::vector<double> &getLnLambda() { return lnLambda; }

    void setDim0(int value) { dim0 = value; }
    void setDim1(int value) { dim1 = value; }
    void setDim1Filtered(int value) { dim1_filtered = value; }
    void setAxName(const std::string &name) { ax_name = name; }
    void setAyName(const std::string &name) { ay_name = name; }

private:
    int dim0;
    int dim1;
    int dim1_filtered;
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> ax_filtered;
    std::vector<double> ay_filtered;
    std::string ax_name;
    std::string ay_name;
    std::vector<double> encoding_qp;
    std::vector<double> lnLambda;

private:
    void parse(const fs::path &path,
                      const std::vector<int> filters,
                      DataSet &out,
                      std::string &ax_name,
                      std::string &ay_name)
    {
        if (fs::exists(path) && fs::is_directory(path))
        {
            for (const auto &entry : fs::directory_iterator(path))
            {
                if (entry.is_regular_file() && entry.path().extension() == ".json")
                {
                    Json::Value root;
                    std::ifstream file(entry.path());

                    if (file.is_open())
                    {
                        file >> root;
                        file.close();

                        out.ax_name = ax_name;
                        out.ay_name = ay_name;

                        for (int fIdx = 0; fIdx < 64; ++fIdx)
                        {
                            const auto &ax = root["frames"][fIdx][out.ax_name];
                            const auto &ay = root["frames"][fIdx][out.ay_name];

                            for (int qp = 22; qp < 52; ++qp)
                            {
                                out.encoding_qp.push_back(qp);
                            }

                            int filterIdx = 0;
                            for (const auto &elem : ax)
                            {
                                out.ax.push_back(elem.asDouble());
                                if (std::find(filters.begin(), filters.end(), filterIdx) != filters.end())
                                {
                                    out.ax_filtered.push_back(elem.asDouble());
                                }
                                ++filterIdx;
                            }

                            filterIdx = 0;
                            for (const auto &elem : ay)
                            {
                                double val = 0;
                                if (ay_name == "vmaf")
                                    val = (100.0 - elem.asDouble()) / 100.0;
                                else
                                    val = elem.asDouble();
                                out.ay.push_back(val);
                                if (std::find(filters.begin(), filters.end(), filterIdx) != filters.end())
                                {
                                    double val = 0;
                                    if (ay_name == "vmaf")
                                        val = (100.0 - elem.asDouble()) / 100.0;
                                    else
                                        val = elem.asDouble();
                                    out.ay_filtered.push_back(val);
                                }
                                ++filterIdx;
                            }
                        }
                    }
                    else
                    {
                        std::cout << "Failed to open JSON file: " << entry.path() << std::endl;
                    }
                }
            }

            out.dim0 = out.ax.size() / 30;
            out.dim1 = 30;
            out.dim1_filtered = filters.size();
        }
        else
        {
            std::cout << "Provided path is not a directory or does not exist." << std::endl;
        }
    }
};

int DataSet::parse_json_data(const char *path,
                             const std::vector<int> filters,
                             DataSet &out,
                             std::string &ax_name,
                             std::string &ay_name)
{
    std::ifstream f(path);
    Json::Value root;

    if (!f.is_open())
    {
        std::cerr << "Error opening file\n";
        return 1;
    }

    f >> root;
    f.close();

    const auto &ax = root[ax_name];
    const auto &ay = root[ay_name];

    out.ax_name = ax_name;
    out.ay_name = ay_name;

    // const auto &ax = root[out.ax_name];
    // const auto &ay = root[out.ay_name];
    const auto &qp = root["qp"];

    out.dim0 = ax.size();
    out.dim1 = ax[0].size();

    out.dim1_filtered = filters.size();

    out.ax.clear();
    out.ax.reserve(out.dim0 * out.dim1);
    out.ay.clear();
    out.ay.reserve(out.dim0 * out.dim1);
    out.ax_filtered.clear();
    out.ax_filtered.reserve(out.dim0 * out.dim1_filtered);
    out.ay_filtered.clear();
    out.ay_filtered.reserve(out.dim0 * out.dim1_filtered);

    out.encoding_qp.clear();
    out.encoding_qp.reserve(out.dim0 * out.dim1);
    out.lnLambda.clear();
    out.lnLambda.reserve(out.dim0 * out.dim1);

    for (const auto &row : qp)
    {
        for (const auto &elem : row)
        {
            out.encoding_qp.push_back(elem.asInt());
        }
    }

    for (const auto &row : ax)
    {
        for (const auto &elem : row)
        {
            out.ax.push_back(elem.asDouble());
        }
        for (int i : filters)
        {
            out.ax_filtered.push_back(row[i].asDouble());
        }
    }

    for (const auto &row : ay)
    {
        for (const auto &elem : row)
        {
            double val = 0;
            if (ay_name == "vmaf")
                val = (100.0 - elem.asDouble()) / 100.0;
            else
                val = elem.asDouble();
            out.ay.push_back(val);
        }
        for (int i : filters)
        {
            double val = 0;
            if (ay_name == "vmaf")
                val = (100.0 - row[i].asDouble()) / 100.0;
            else
                val = row[i].asDouble();
            out.ay_filtered.push_back(val);
        }
    }

    return 0;
}

void DataSet::write_json_data(const char *path, const double *ax, const double *ay, int n)
{
    std::ofstream f(path);

    Json::Value root;
    Json::Value x = Json::Value{Json::arrayValue};
    Json::Value y = Json::Value{Json::arrayValue};

    for (int i = 0; i < n; ++i)
    {
        x.append(ax[i]);
        y.append(ay[i]);
    }

    root["ax"] = x;
    root["ay"] = y;
    f << root;
}
