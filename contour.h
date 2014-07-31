#ifndef CONTOUR_H
#define CONTOUR_H

#include <vec2.h>
#include <linkedlist.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace qimp {

class Contour {

  public:
    Contour();
    ~Contour();

    void add(qm::Vec2i& point);
    void add(int i, int j);
    void addFront(int i, int j);
    void merge(Contour& contour);
    qds::LinkedList<qm::Vec2i>* getPoints();
    int getSize();
    void clear();

    qm::Vec2i getBarycenter() { return barycenter; }

    void findFingertips(qds::LinkedList<qm::Vec2i>& fingertips, int K, float theta, cv::Mat& handRegion, bool debug, int J, cv::Mat& destination);

  private:
    qds::LinkedList<qm::Vec2i>* points;
    qm::Vec2i barycenter;
    int size;

};

}

#endif // CONTOUR_H
