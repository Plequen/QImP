#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gaussianmixturemodel.h"
#include "histogramshistory.h"


namespace qimp {

class Segmentation {

  public:
    Segmentation(float probaBlendFactor = 0.9);
    ~Segmentation();

    bool loadColorDistributions(const std::string& skinFilename, const std::string& nonSkinFilename, const std::string& skinLookUpTableFilename, const std::string& nonSkinLookUpTableFilename);
    bool loadHandMask(const std::string& filename);
    bool initHandMask(const cv::Mat& image);

    void handRegionViaGMM(const cv::Mat& image, cv::Mat& destination, bool fScreenShot = false);
    void handRegionViaGMM2(const cv::Mat& image, cv::Mat& destination, bool fScreenShot = false); // for tests
    void computeHandRegion(const cv::Mat& image, cv::Mat& destination, bool fScreenShot = false);

    cv::Mat& getHandRegion() { return binaryHandRegion; }
    cv::Mat& getHandMask() { return handMask; }

    void learnYCrCbColor(bool learn);
    void learnYCrCbColor(bool learn, cv::Mat& mask);


  private:
    GaussianMixtureModel skinColor;
    GaussianMixtureModel nonSkinColor;

    HistogramsHistory* histograms;

    cv::Mat binaryHandRegion;
    cv::Mat imageGradient;
    cv::Mat handMask;
    cv::Mat imageYCrCb;

    bool fLearned;
    float probaBlendFactor;

};

}

#endif // SEGMENTATION_H
