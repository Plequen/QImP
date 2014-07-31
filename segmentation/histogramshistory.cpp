#include "histogramshistory.h"

using namespace qimp;

HistogramsHistory::HistogramsHistory() {
  histogramsNumber = 0;
  histogramIndex = 0;
  histograms = NULL;
  totalPixels = 0;
  usedSlots = 0;
}

HistogramsHistory::HistogramsHistory(int histogramsNumber, int binsNumber) {
  init(histogramsNumber, binsNumber);
}

void HistogramsHistory::reset(int histogramsNumber, int binsNumber) {
  delete[] histograms;
  init(histogramsNumber, binsNumber);
}

HistogramsHistory::~HistogramsHistory() {
  delete[] histograms;
}

void HistogramsHistory::init(int histogramsNumber, int binsNumber) {
  this->histogramsNumber = histogramsNumber;
  histogramIndex = 0;
  totalPixels = 0;
  usedSlots = 0;
  histograms = new ColorHistogram[histogramsNumber];
  for (int i = 0 ; i < histogramsNumber ; i++)
    histograms[i].init(binsNumber);
}

void HistogramsHistory::reset() {
  histogramIndex = 0;
  totalPixels = 0;
  usedSlots = 0;
}

void HistogramsHistory::learnFromFrame(const cv::Mat& frame, const cv::Mat& mask) {
  totalPixels -= histograms[histogramIndex].getTotalPixelsNumber();
  histograms[histogramIndex].learnHistogram(frame, mask);
  totalPixels += histograms[histogramIndex].getTotalPixelsNumber();
  histogramIndex = (histogramIndex + 1) % histogramsNumber;
  if (usedSlots < histogramsNumber)
    usedSlots++;
}

float HistogramsHistory::getProbability(int value1, int value2, int value3) {
  if (totalPixels == 0)
    return 0.f;

  int pixelsNumber = 0;
  for (int i = 0 ; i < usedSlots ; i++)
    pixelsNumber += histograms[i].getPixelsNumberAt(value1, value2, value3);

  float probability = ((float) pixelsNumber) / ((float) totalPixels);

  return probability;
}
