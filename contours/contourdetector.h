#ifndef CONTOURDETECTOR_H
#define CONTOURDETECTOR_H

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "contour.h"


namespace qimp {

class ContourDetector {

  public:
    ContourDetector();

    void contourImageFromBinary(cv::Mat& source, cv::Mat& destination);
    void contourImageFromGray(cv::Mat& source, cv::Mat& destination, int diff);
    void extractContour(uchar* sourceData, Contour* contour, int i, int j, bool** visited, int patchSize, int rows, int cols, bool forward);
    void extractContours(cv::Mat& source, std::vector<Contour*>& contours, int patchSize, int contourMinSize);
    void extractContours(cv::Mat& source, std::vector<Contour*>& contours, int patchSize, int contourMinSize, cv::Mat& destinationImage);
    void extract(cv::Mat& source, std::vector<Contour*>& contours, int patchSize, int contourMinSize, bool showStepsResults = false);

    void maskFromContours(std::vector<Contour*>& contours, cv::Mat& destination);

    inline cv::Mat getContoursImage() { return contoursImage; }
    inline cv::Mat getExtractedContoursImage() { return extractedContoursImage; }

  private:
    cv::Mat contoursImage;
    cv::Mat extractedContoursImage;

};

}

#endif // CONTOURDETECTOR_H
