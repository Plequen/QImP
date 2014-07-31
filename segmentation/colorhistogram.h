#ifndef COLORHISTOGRAM_H
#define COLORHISTOGRAM_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace qimp {

class ColorHistogram {

  public:
    ColorHistogram();
    ColorHistogram(int binsNumber);
    ~ColorHistogram();

    void init(int binsNumber);

    inline int getTotalPixelsNumber() const { return pixelsNumber; }
    void learnHistogram(const cv::Mat& frame, const cv::Mat& mask);
    int getPixelsNumberAt(int value1, int value2, int value3);

  private:
    int pixelsNumber;
    int binsNumber;
    int*** pixels;

};

}

#endif // COLORHISTOGRAM_H
