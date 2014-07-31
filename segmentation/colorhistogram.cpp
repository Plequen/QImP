#include "colorhistogram.h"

using namespace qimp;

ColorHistogram::ColorHistogram() {
  pixelsNumber = 0;
  binsNumber = 0;
  pixels = NULL;
}

ColorHistogram::ColorHistogram(int binsNumber) {
  pixelsNumber = 0;
  init(binsNumber);
}

ColorHistogram::~ColorHistogram() {
  for (int i = 0 ; i < binsNumber ; i++) {
    for (int j = 0 ; j < binsNumber ; j++)
      delete[] pixels[i][j];
    delete[] pixels[i];
  }
  delete[] pixels;
}

void ColorHistogram::init(int binsNumber) {
  this->binsNumber = binsNumber;
  pixels = new int**[binsNumber];
  for (int i = 0 ; i < binsNumber ; i++) {
    pixels[i] = new int*[binsNumber];
    for (int j = 0 ; j < binsNumber ; j++)
      pixels[i][j] = new int[binsNumber];
  }

}

void ColorHistogram::learnHistogram(const cv::Mat& frame, const cv::Mat& mask) {
  // Reset the histogram
  for (int i = 0 ; i < binsNumber ; i++)
    for (int j = 0 ; j < binsNumber ; j++)
      for (int k = 0 ; k < binsNumber ; k++)
        pixels[i][j][k] = 0;
  pixelsNumber = 0;

  // Learn the histogram from the frame
  uchar* maskData = (uchar*) mask.data;
  uchar* frameData = (uchar*) frame.data;
  for (int i = 0 ; i < mask.rows ; i++) {
    for (int j = 0 ; j < mask.cols ; j++) {
      if (maskData[i*mask.step + j] == 255) {
        uchar value1 = frameData[i*frame.step + 3*j];
        uchar value2 = frameData[i*frame.step + 3*j + 1];
        uchar value3 = frameData[i*frame.step + 3*j + 2];
        pixels[value1/binsNumber][value2/binsNumber][value3/binsNumber]++;
        pixelsNumber++;
      }
    }
  }
}

int ColorHistogram::getPixelsNumberAt(int value1, int value2, int value3) {
  return pixels[value1/binsNumber][value2/binsNumber][value3/binsNumber];
}
