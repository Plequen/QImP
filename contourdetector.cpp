#include "contourdetector.h"

using namespace qimp;
using namespace std;
using namespace cv;

ContourDetector::ContourDetector() { }

void ContourDetector::contourImageFromBinary(cv::Mat& source, cv::Mat& destination) {
  destination = Mat::zeros(source.rows, source.cols, CV_8UC1);
  bool prevYOut[source.cols];
  for (int j = 0 ; j < source.cols ; j++)
    prevYOut[j] = true;
  bool prevOut;
  uchar* rowI;
  uchar* outRowI;
  uchar val;
  bool currentIn;
  for (int i = 0 ; i < source.rows ; i++) {
    rowI = source.ptr<uchar>(i);
    outRowI = destination.ptr<uchar>(i);
    prevOut = true;
    for (int j = 0 ; j < source.cols ; j++) {
      val = rowI[j];
      currentIn = val != 0;
      if (prevOut && currentIn)
        outRowI[j] = (uchar) 255;
      else if (!prevOut && !currentIn)
        outRowI[j-1] = (uchar) 255;
      prevOut = !currentIn;
      if (prevYOut[j] && currentIn)
        outRowI[j] = (uchar) 255;
      else if (!prevYOut[j] && !currentIn)
        outRowI[j] = (uchar) 255;
      prevYOut[j] = !currentIn;
    }
    if (!prevOut)
      outRowI[source.cols-1] = (uchar) 255;
  }
  outRowI = destination.ptr<uchar>(source.rows-1);
  for (int j = 0 ; j < source.cols ; j++) {
    if (!prevYOut[j])
      outRowI[j] = (uchar) 255;
  }
}

void ContourDetector::contourImageFromGray(cv::Mat& source, cv::Mat& destination, int diff) {
  destination = Mat::zeros(source.rows, source.cols, source.type());
  uchar prevY[source.cols];
  for (int j = 0 ; j < source.cols ; j++)
    prevY[j] = 0;
  uchar** outRows = new uchar*[source.rows];
  for (int i = 0 ; i < source.rows ; i++)
    outRows[i] = destination.ptr<uchar>(i);
  uchar* rowI;
  for (int i = 0 ; i < source.rows ; i++) {
    rowI = source.ptr<uchar>(i);
    uchar prev = rowI[0];
    for (int j = 1 ; j < source.cols ; j++) {
      uchar val = rowI[j];
      bool changed = val > prev ? val - prev > diff : prev - val > diff;
      if (changed) {
        outRows[i][j] = (uchar) val;
        if (j > 0)
          outRows[i][j-1] = (uchar) prev;
        prev = val;
      }
      else if (i != 0) {
        bool changedY = val > prevY[j] ? val - prevY[j] > diff : prevY[j] - val > diff;
        if (changedY) {
          outRows[i][j] = (uchar) val;
          outRows[i-1][j] = (uchar) prevY[j];
          prevY[j] = val;
        }
      }
      //prev = val;
      //prevY[j] = val;
    }
  }
}

void ContourDetector::extractContour(uchar* sourceData, Contour* contour, int i, int j, bool** visited, int patchSize, int rows, int cols, bool forward) {
  if (forward)
    contour->add(i, j);
  else
    contour->addFront(i, j);

  bool forwardDir = forward;
  int pathsNumber = 0;
  int lastPathX = 0, lastPathY = 0;
  int x, y;
  for (int k = -patchSize ; k <= patchSize ; k++) {
    for (int l = -patchSize ; l <= patchSize ; l++) {
      y = i + k;
      x = j + l;
      if (y >= 0 && y < rows && x >= 0 && x < cols) {
        if (!visited[y][x] && sourceData[y*cols + x] != 0) {
          lastPathY = y;
          lastPathX = x;
          pathsNumber++;
        }
      }
    }
  }
  if (pathsNumber == 1) {
    visited[lastPathY][lastPathX] = true;
    extractContour(sourceData, contour, lastPathY, lastPathX, visited, patchSize, rows, cols, forwardDir);
  }
  else if (pathsNumber > 1) {
    Contour* contourKept = NULL;
    bool alreadyContour = false;
    for (int k = -patchSize ; k <= patchSize ; k++) {
      for (int l = -patchSize ; l <= patchSize ; l++) {
        y = i + k;
        x = j + l;
        if (y >= 0 && y < rows && x >= 0 && x < cols) {
          if (!visited[y][x] && sourceData[y*cols + x] != 0) {
            visited[y][x] = true;
            Contour* tempContour = new Contour();
            extractContour(sourceData, tempContour, y, x, visited, patchSize, rows, cols, forwardDir);
            if (tempContour->getSize() >= 200) {
              contour->merge(*tempContour);
              forwardDir = !forwardDir;
              delete contourKept;
              contourKept = NULL;
              alreadyContour = true;
              delete tempContour;
            }
            else if (!alreadyContour && (contourKept == NULL || tempContour->getSize() > contourKept->getSize()))
              contourKept = tempContour;
            else
              delete tempContour;
          }
        }
      }
    }
    if (!alreadyContour) {
      contour->merge(*contourKept);
      delete contourKept;
    }
  }
}

void ContourDetector::extractContours(cv::Mat& source, std::vector<Contour*>& contours, int patchSize, int contourMinSize) {
  // Init the visited pixels matrix
  bool** visited = new bool*[source.rows];
  for (int i = 0 ; i < source.rows ; i++) {
    visited[i] = new bool[source.cols];
    for (int j = 0 ; j < source.cols ; j++)
      visited[i][j] = false;
  }
  // Extract the contours and store them into the contours vector
  uchar* sourceData = (uchar*) source.data;
  uchar val;
  Contour* newContour;
  for (int i = 0 ; i < source.rows ; i++) {
    for (int j = 0 ; j < source.cols ; j++) {
      val = sourceData[i*source.step + j];
      if (!visited[i][j] && val != 0) {
        visited[i][j] = true;
        // Extract a new contour starting from this pixel
        newContour = new Contour();
        extractContour(sourceData, newContour, i, j, visited, patchSize, source.rows, source.cols, true);
        if (newContour->getSize() >= contourMinSize)
          contours.push_back(newContour);
      }
    }
  }
  delete[] visited;
}

void ContourDetector::extractContours(cv::Mat& source, std::vector<Contour*>& contours, int patchSize, int contourMinSize, cv::Mat& destinationImage) {
  destinationImage = Mat::zeros(source.rows, source.cols, CV_8UC3);
  extractContours(source, contours, patchSize, contourMinSize);
  vector<Contour*>::iterator i = contours.begin();
  int contourIndex = 0;
  while (i != contours.end()) {
    qds::LinkedList<qm::Vec2i>* points = (*i)->getPoints();
    points->startBrowse();
    qm::Vec2i point;
    while (points->browse(point)) {
      destinationImage.at<Vec3b>(point[0], point[1])[contourIndex % 3] = (uchar) 255;
    }
    contourIndex++;
    i++;
  }
}

void ContourDetector::extract(cv::Mat& source, std::vector<Contour*>& contours, int patchSize, int contourMinSize, bool showStepsResults) {
  contourImageFromBinary(source, contoursImage);
  if (showStepsResults) {
    //imshow("Contours image", contoursImage);
    extractContours(contoursImage, contours, patchSize, contourMinSize, extractedContoursImage);
    imshow("Extracted contours", extractedContoursImage);
  }
  else
    extractContours(contoursImage, contours, patchSize, contourMinSize);
}

void ContourDetector::maskFromContours(std::vector<Contour*>& contours, cv::Mat& destination) {
  vector<Contour*>::iterator i = contours.begin();
  while (i != contours.end()) {
    qds::LinkedList<qm::Vec2i>* points = (*i)->getPoints();
    points->startBrowse();
    qm::Vec2i point;
    while (points->browse(point))
      destination.at<uchar>(point[0], point[1]) = (uchar) 255;
    i++;
  }
  uchar* imageRow;
  bool yIn[destination.cols];
  for (int j = 0 ; j < destination.cols ; j++) {
   yIn[j] = false;
  }
  for (int i = 0 ; i < destination.rows ; i++) {
    imageRow = destination.ptr<uchar>(i);
    bool in = false;
    bool prevOut = true;
    for (int j = 0 ; j < destination.cols ; j++) {
      if (imageRow[j] == 255 && !in) {
        in = true;
      }
      else if (imageRow[j] == 255 && in && prevOut) {
        in = false;
      }
      if (imageRow[j] == 255) {
        prevOut = false;
      }
      else
        prevOut = true;
      if (in || yIn[j])
        imageRow[j] = 255;
    }
  }
}
