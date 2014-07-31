#ifndef HISTOGRAMSHISTORY_H
#define HISTOGRAMSHISTORY_H

#include <iostream>

#include "colorhistogram.h"


namespace qimp {

class HistogramsHistory {

  public:
    HistogramsHistory();
    HistogramsHistory(int histogramsNumber, int binsNumber);
    ~HistogramsHistory();

    void init(int histogramsNumber, int binsNumber);
    void reset();
    void reset(int histogramsNumber, int binsNumber);
    void learnFromFrame(const cv::Mat& frame, const cv::Mat& mask);
    float getProbability(int value1, int value2, int value3);

  private:
    int histogramsNumber;
    int histogramIndex;
    ColorHistogram* histograms;
    int totalPixels;
    int usedSlots;

};

}

#endif // HISTOGRAMSHISTORY_H
