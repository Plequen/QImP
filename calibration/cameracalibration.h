#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


namespace qimp {

class CameraCalibration {

  public:
    CameraCalibration(unsigned int minFrames = 5);
    CameraCalibration(int chessRows, int chessCols, float chessLength, unsigned int minFrames = 5);

    bool loadCalibration(const std::string& filename);

    bool feedColorFrame(cv::Mat& frame, bool display = false);
    bool feedGrayFrame(cv::Mat& grayFrame, bool display = false);
    bool enoughFrames() const { return imagePoints.size() >= (unsigned int) minFrames; }
    void reset();

    void calibrate();
    bool saveCalibration(const std::string& filename);

    bool isCalibrated() const { return calibrated; }

    cv::Mat getCameraMatrix() const { return cameraMatrix; }
    cv::Mat getDistorsionCoeffs() const { return distorsionCoeffs; }

    int getChessRows() { return chessRows; }
    int getChessCols() { return chessCols; }


  private:
    int chessRows, chessCols;
    float chessLength;
    std::vector<cv::Point3f> chessCorners;

    bool calibrated;
    int minFrames;
    cv::Size framesSize;
    std::vector<cv::Point2f> pointsBuffer;
    std::vector<std::vector<cv::Point2f> > imagePoints;

    void computeCornersPositions();

    cv::Mat cameraMatrix;
    cv::Mat distorsionCoeffs;

};

}

#endif // CAMERACALIBRATION_H
