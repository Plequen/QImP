#ifndef GAUSSIANMIXTUREMODEL_H
#define GAUSSIANMIXTUREMODEL_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

#include <opencv2/core/core.hpp>

#define MAX_NUM_GAUSSIAN  100


namespace qimp {

class GaussianMixtureModel {

  public:
    GaussianMixtureModel();
    GaussianMixtureModel(int mixtures, int dimensions);
    ~GaussianMixtureModel();

    bool loadFromFile(const std::string& filename);
    void addGaussian(cv::Mat& meanMat, cv::Mat& covMat, float weight);

    float getProbability(const cv::Mat& sample);

    void makeLookUpTable();
    bool saveLookUpTable(const std::string& filename);
    bool loadLookUpTable(const std::string& filename);
    float getProbabilityByLookUp(int r, int g, int b);

    void printLookUpTable();

  private:
    int mixtures;
    int dimensions;

    std::vector<cv::Mat> meanMatrices;
    std::vector<cv::Mat> covMatrices;
    std::vector<cv::Mat> covMatricesI;
    float* weights;

    float*** probability;

};

}

#endif // GAUSSIANMIXTUREMODEL_H
