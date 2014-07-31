#include "cameracalibration.h"

using namespace qimp;
using namespace cv;
using namespace std;

CameraCalibration::CameraCalibration(unsigned int minFrames) {
  chessRows = 0;
  chessCols = 0;
  chessLength = 0.f;
  this->minFrames = minFrames;
  calibrated = false;
}

CameraCalibration::CameraCalibration(int chessRows, int chessCols, float chessLength, unsigned int minFrames) {
  this->chessRows = chessRows;
  this->chessCols = chessCols;
  this->chessLength = chessLength;
  this->minFrames = minFrames;
  calibrated = false;
  computeCornersPositions();
}

void CameraCalibration::computeCornersPositions() {
  chessCorners.clear();
  float squareSize = chessLength / ((float) chessCols);
  for (int i = 0 ; i < chessRows ; i++) {
    for (int j = 0 ; j < chessCols ; j++)
      chessCorners.push_back(Point3f(float(j*squareSize), float(-i*squareSize), 0.f));
  }
}

bool CameraCalibration::loadCalibration(const string& filename) {
  ifstream file(filename.c_str());
  if (!file) {
    cerr << "Could not load the file. " << endl;
    return false;
  }

  cameraMatrix = Mat(3, 3, CV_64F);
  distorsionCoeffs = Mat(8, 1, CV_64F);
  for (int i = 0 ; i < 3 ; i++) {
    for (int j = 0 ; j < 3 ; j++)
      file >> cameraMatrix.at<double>(i, j);
  }
  for (int i = 0 ; i < 8 ; i++)
    file >> distorsionCoeffs.at<double>(i);

  cout << "Calibration loaded from file " << filename << " : " << endl;
  cout << cameraMatrix << endl << distorsionCoeffs << endl;

  file.close();
  return true;
}

bool CameraCalibration::feedColorFrame(Mat& frame, bool display) {
  Mat grayFrame;
  cvtColor(frame, grayFrame, CV_BGR2GRAY);
  bool found = feedGrayFrame(grayFrame, false);
  if (found && display) {
    drawChessboardCorners(frame, Size(chessCols, chessRows), Mat(pointsBuffer), true);
    imshow("Camera calibration", frame);
  }
  return found;
}

bool CameraCalibration::feedGrayFrame(Mat& grayFrame, bool display) {
  framesSize.width = grayFrame.cols;
  framesSize.height = grayFrame.rows;
  // detect the chessboard
  bool chessboardFound = false;
  pointsBuffer.clear();
  chessboardFound = findChessboardCorners(
    grayFrame,
    Size(chessCols, chessRows),
    pointsBuffer,
    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE
  );
  if (chessboardFound) {
    // improve the found corners' coordinate accuracy
    cornerSubPix(
      grayFrame, pointsBuffer, Size(11,11),
      Size(-1,-1),
      TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1)
    );
    imagePoints.push_back(pointsBuffer);
    if (display) {
      drawChessboardCorners(grayFrame, Size(chessCols, chessRows), Mat(pointsBuffer), chessboardFound);
      imshow("Camera calibration", grayFrame);
    }
    return true;
  }
  return false;
}

void CameraCalibration::reset() {
  imagePoints.clear();
  calibrated = false;
}

void CameraCalibration::calibrate() {
  if (imagePoints.size() == 0)
    return;

  vector<vector<Point3f> > objectPoints(0);
  objectPoints.resize(imagePoints.size(), chessCorners);

  cameraMatrix = Mat::eye(3, 3, CV_64F); // identity matrix
  cameraMatrix.at<double>(0,0) = 1.0; // fixed ratio
  distorsionCoeffs = Mat::zeros(8, 1, CV_64F);

  // calibrate the camera
  vector<Mat> rvecs, tvecs;
  double rms = calibrateCamera(
    objectPoints, imagePoints, framesSize,
    cameraMatrix, distorsionCoeffs,
    rvecs, tvecs, CV_CALIB_FIX_ASPECT_RATIO|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5
  );

  calibrated = true;
  cout << "Calibration error: " << rms << endl;
}

bool CameraCalibration::saveCalibration(const std::string& filename) {
  if (!calibrated) {
    cerr << "Camera not calibrated, cannot save the calibration." << endl;
    return false;
  }
  ofstream file(filename.c_str(), ios::out);
  if (!file) {
    cerr << "Could not load the file. " << endl;
    return false;
  }
  for (int i = 0 ; i < 3 ; i++) {
    for (int j = 0 ; j < 3 ; j++)
      file << cameraMatrix.at<double>(i, j) << " ";
  }
  file << endl;
  for (int i = 0 ; i < distorsionCoeffs.rows ; i++) {
    file << distorsionCoeffs.at<double>(i) << " ";
  }
  for (int i = distorsionCoeffs.rows ; i < 8 ; i++) {
    file << (double) 0 << " ";
  }
  file << endl;
  file.close();

  return true;
}
