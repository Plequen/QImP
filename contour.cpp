#include "contour.h"

#include <map>

using namespace qimp;
using namespace std;

Contour::Contour() {
  size = 0;
  points = new qds::LinkedList<qm::Vec2i>();
}

Contour::~Contour() {
  delete points;
}

void Contour::add(qm::Vec2i& point) {
  points->pushBack(point);
  size++;
}

void Contour::add(int i, int j) {
  points->pushBack(qm::Vec2i(i, j));
  size++;
}

void Contour::addFront(int i, int j) {
  points->pushFront(qm::Vec2i(i, j));
  size++;
}

void Contour::merge(Contour& contour) {
  size += contour.getSize();
  points->appendBack(contour.getPoints());
}

qds::LinkedList<qm::Vec2i>* Contour::getPoints() {
  return points;
}

int Contour::getSize() {
  return size;
}

void Contour::clear() {
  size = 0;
  points->clear();
}

int mod(int a, int b) {
  return (a % b + b) % b;
}

void drawPointOnImage(qm::Vec2i& point, cv::Mat& image, uchar r, uchar g, uchar b, int size) {
  for (int p = -size ; p <= size ; p++) {
    for (int l = -size ; l <= size ; l++) {
      int x = point[1]+l;
      int y = point[0]+p;
      if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
        image.at<cv::Vec3b>(y,x)[0] = (uchar) b;
        image.at<cv::Vec3b>(y,x)[1] = (uchar) g;
        image.at<cv::Vec3b>(y,x)[2] = (uchar) r;
      }
    }
  }
}

void Contour::findFingertips(qds::LinkedList<qm::Vec2i>& fingertips, int K, float theta, cv::Mat& handRegion, bool debug, int J, cv::Mat& destination) {
  uchar* handRegionData = (uchar*) handRegion.data;
  barycenter = qm::Vec2i(0, 0);
  qm::Vec2i* pointsArray = points->toArray();
  qm::Vec2i currentFingerTip(0, 0);
  int fingerTipCandidates = 0;
  int lastCandidateIndex = -1;
  int firstCandidateIndex = -1;
  int currentFirst = -1, currentLast = -1;
  int beforeIndex, afterIndex;
  qm::Vec2i current, before, after, bc, ac, mean;

  int highCurvatureChanges = 0;
  float highCurvatureChangeThreshold = 20.f;
  float lastAngle = -1.f;

  for (int j = 0 ; j < size ; j++) {
    current = pointsArray[j];
    barycenter += current;
    beforeIndex = mod(j - K, size);
    afterIndex = mod(j + K, size);
    before = pointsArray[beforeIndex];
    after = pointsArray[afterIndex];
    bc = before - current;
    ac = after - current;
    if (bc.squaredLength() < 10000 && ac.squaredLength() < 10000) {
      int dotProduct = qm::Vec2i::dotProduct(bc, ac);
      double cosAngle = ((double) dotProduct) / ((double) bc.getLength() * ac.getLength());
      float angle = acos(cosAngle) * 180 / M_PI;
      if (lastAngle > 0) {
        float angleDiff = angle - lastAngle;
        if (angleDiff < 0.f)
          angleDiff = -angleDiff;
        if (angleDiff >= highCurvatureChangeThreshold)
          highCurvatureChanges++;
      }
      lastAngle = angle;
      //if (mod(J, size) == j)
      //  cout << "Position : " << j << ", angle : " << angle << endl;
      if (angle < theta) {
        if (lastCandidateIndex == -1 || abs(lastCandidateIndex - j) > 5 ) {
          if (lastCandidateIndex != -1) {
            // Remove false positives
            int meanIndex = (currentFirst + currentLast) / 2;
            qm::Vec2i finger = (pointsArray[mod(meanIndex-K, size)] + pointsArray[mod(meanIndex+K, size)]) / 2;
            drawPointOnImage(finger, destination, 255, 0, 255, 0);
            //fingertips.pushBack(finger);
            if (handRegionData[finger[0]*handRegion.cols + finger[1]] == 255) { // (before+after)/2 is in hand region
              int distMinFromEdge = 10;
              qm::Vec2i newFinger = currentFingerTip / fingerTipCandidates;
              if (true || handRegionData[newFinger[0]*handRegion.cols + newFinger[1]] == 255) {
                if (newFinger[0] >= distMinFromEdge && newFinger[1] >= distMinFromEdge && newFinger[0] <= handRegion.rows - distMinFromEdge && newFinger[1] <= handRegion.cols - distMinFromEdge) {
                  fingertips.pushBack(newFinger);
                }
              }
            }
          }
          currentFingerTip = current;
          fingerTipCandidates = 1;
          currentFirst = j;
        }
        else {
          // filter at the finger candidate level
          //mean = (before + after) / 2;
          //if (handRegionData[mean[0]*handRegion.cols + mean[1]] == 255) {
            currentFingerTip += current;
            fingerTipCandidates++;
            //fingertips.pushBack(current);
          //}
        }
        if (firstCandidateIndex == -1)
          firstCandidateIndex = j;
        lastCandidateIndex = j;
        currentLast = j;
      }
    }
    if (debug) {
      if (j == mod(J, size)) {
        drawPointOnImage(current, destination, 255, 0, 0, 2);
        drawPointOnImage(before, destination, 0, 255, 0, 2);
        drawPointOnImage(after, destination, 255, 0, 255, 2);
      }
    }
  }
  barycenter /= size;

  //cout << "High curvature changes ratio : " << (((float) highCurvatureChanges) / size) << endl;

  if (lastCandidateIndex != -1) {
    if (fingertips.getSize() == 0 || mod(firstCandidateIndex - lastCandidateIndex, size) > 5)
      fingertips.pushBack(currentFingerTip / fingerTipCandidates);
    else {
      qm::Vec2i lastPartFinger = currentFingerTip / fingerTipCandidates;
      qm::Vec2i realFinger = (fingertips.getFront()->getElement() + lastPartFinger) / 2;
      fingertips.getFront()->setElement(realFinger);
    }
  }
  if (highCurvatureChanges < 5) {
    // Sort fingertips
    int fingertipsNumber = fingertips.getSize();
    if (fingertipsNumber == 5) {
      fingertips.startBrowse();
      qm::Vec2i fingertip[fingertipsNumber];
      int f = 0;
      int thumbIndex = -1;
      while (fingertips.browse(fingertip[f]))
        f++;
        int maxDistToOthers = 0;
      for (int k = 0 ; k < fingertipsNumber ; k++) {
        int distToOthers = 0;
        for (int l = 0 ; l < fingertipsNumber ; l++) {
          if (k != l) {
            distToOthers += (fingertip[k] - fingertip[l]).squaredLength();
          }
        }
        if (distToOthers > maxDistToOthers) {
          maxDistToOthers = distToOthers;
          thumbIndex = k;
        }
      }
      map<int, int> distancesToThumb;
      for (int l = 0 ; l < fingertipsNumber ; l++) {
        if (l != thumbIndex) {
          distancesToThumb.insert(pair<int, int>((fingertip[thumbIndex] - fingertip[l]).squaredLength(),l));
        }
      }
      fingertips.clear();
      fingertips.pushBack(fingertip[thumbIndex]);
      drawPointOnImage(fingertip[thumbIndex], destination, 255, 255, 255, 5);
      map<int, int>::iterator it;
      int l = 0;
      for (it = distancesToThumb.begin() ; it != distancesToThumb.end(); it++) {
        fingertips.pushBack(fingertip[it->second]);
        //cout << "finger " << l << " : " << it->first << endl;
        if (l == 0)
          drawPointOnImage(fingertip[it->second], destination, 255, 0, 0, 5);
        else if (l == 1)
          drawPointOnImage(fingertip[it->second], destination, 0, 255, 0, 5);
        else if (l == 2)
          drawPointOnImage(fingertip[it->second], destination, 0, 0, 255, 5);
        else if (l == 3)
          drawPointOnImage(fingertip[it->second], destination, 255, 255, 0, 5);
        l++;
      }
    }
    else {
      // Draw fingertips
      fingertips.startBrowse();
      qm::Vec2i fingertip;
      while (fingertips.browse(fingertip))
        drawPointOnImage(fingertip, destination, 128, 128, 128, 5);
    }

    // Draw barycenter
    drawPointOnImage(barycenter, destination, 0, 0, 0, 4);
  }
  else
    drawPointOnImage(barycenter, destination, 255, 0, 0, 4);
  delete pointsArray;
}

