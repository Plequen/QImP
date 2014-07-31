#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>

#include <linkedlist.h>
#include <vec2.h>

#include "segmentation/segmentation.h"
#include "contours/contour.h"
#include "contours/contourdetector.h"

using namespace std;
using namespace cv;
using namespace qimp;


string IMAGES_FOLDER, VIDEOS_FOLDER, DATA_FOLDER, PROJECT_FOLDER, OUTPUT_FOLDER;

Mat handImage, medianBlurred, filteredHand, greyHand;
Mat extractedContour;
Mat filteredRegion;
vector<Contour*> frameContours;
Mat laplacianInput;
Mat frame, frameFilteredSegmentation;

int cannyThreasholdRatio = 3;

const int medianSliderMax = 50;
int medianSlider = 1;

const int laplacianSliderMax = 50;
int laplacianSlider = 1;

const int fingersSliderMax = 3000;
int fingersSlider = 1;


bool initFromConfigFile(const std::string& filename) {
  ifstream file(filename.c_str());
  if (!file) {
    cerr << "Could not load the config file. " << endl;
    return false;
  }
  string line, head;
  while (getline(file, line, '\n')) {
    stringstream lineStream(line);
    lineStream >> head;
    if (head.compare("IMAGES") == 0)
      lineStream >> IMAGES_FOLDER;
    else if (head.compare("VIDEOS") == 0)
      lineStream >> VIDEOS_FOLDER;
    else if (head.compare("DATA") == 0)
      lineStream >> DATA_FOLDER;
    else if (head.compare("PROJECT") == 0)
      lineStream >> PROJECT_FOLDER;
    else if (head.compare("OUTPUT") == 0)
      lineStream >> OUTPUT_FOLDER;
  }
  file.close();
  return true;
}

void conv2(Mat& sourceImage, Mat& destinationImage, int kernelSize) {
  Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float) (kernelSize * kernelSize);
  filter2D(sourceImage, destinationImage, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
}

//
// Morphological operations
//
void erosion(Mat& src, Mat& dst, int erosionSize) {
  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*erosionSize + 1, 2*erosionSize + 1), Point(erosionSize, erosionSize));
  erode(src, dst, element);
}

void dilatation(Mat& src, Mat& dst, int size) {
  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*size + 1, 2*size + 1), Point(size, size));
  dilate(src, dst, element);
}

void openingB(Mat& source, Mat& destination, int size) {
  Mat eroded;
  erosion(source, eroded, size);
  dilatation(eroded, destination, size);
}

void closingB(Mat& source, Mat& destination, int size) {
  Mat dilated;
  dilatation(source, dilated, size);
  erosion(dilated, destination, size);
}

void cannyThreshold(Mat& source, Mat& destination, int lowThreshold, int kernelSize) {
  Mat detectedEdges;
  // Reduce noise with a 3x3 kernel
  // blur(source, detectedEdges, Size(3, 3));
  // imshow("blured", detectedEdges);

  // Canny detector
  source.copyTo(detectedEdges);
  Canny(detectedEdges, detectedEdges, lowThreshold, lowThreshold*cannyThreasholdRatio, kernelSize);
  imshow("canny detector", detectedEdges);

  // Use Canny's output as a mask
  destination = Scalar::all(0);
  source.copyTo(destination, detectedEdges);
}

void onMedianTrackbar(int, void*) {
  cout << medianSlider << endl;
  int size = medianSlider;
  if (size % 2 == 0)
    size++;
  medianBlur(handImage, medianBlurred, size);
  imshow("Median blur", medianBlurred);
  Mat grey;
  cvtColor(medianBlurred, grey, CV_BGR2GRAY);
  imshow("Grey median", grey);

  Mat abs_dst;
  int lsize = laplacianSlider;
  if (lsize % 2 == 0)
    lsize++;
  Laplacian( grey, filteredHand, -1, lsize, 1, 0, BORDER_DEFAULT );
  convertScaleAbs( filteredHand, abs_dst );

  imshow("Laplacian abs", abs_dst);

  //cannyThreshold(grey, filteredHand, 20, 3);
  //imshow("Canny", filteredHand);
}

void onLaplacianTrackbar(int, void*) {
  Mat laplacian;
   int size = laplacianSlider;
  if (size % 2 == 0)
    size++;
  Mat grey;
  blur(laplacianInput, grey, Size(3, 3));
  //medianBlur(laplacian, grey, 3);
  Laplacian( grey, laplacian, -1, size, 1, 0, BORDER_DEFAULT );
  Mat abs_dst;
  convertScaleAbs( laplacian, abs_dst );
  //threshold(abs_dst, laplacian, 10, 255, 0);
 // medianBlur(abs_dst, laplacian, 3);
  Mat thres;
  threshold(abs_dst, thres, 126, 255, 0);
  imshow("Laplacian threshold", thres);

  // ouverture : remove small detached pieces
  Mat ouv, ouv2;
  erosion(thres, ouv, 1);
  imshow("ouv erosion", ouv);
  dilatation(ouv, ouv2, 1);
  imshow("ouv dilatation", ouv2);

  // fermeture : close edges
  Mat fem, fem2;
  dilatation(ouv2, fem, 2);
  imshow("fem dilatation", fem);
  erosion(fem, fem2, 2);
  imshow("fem erosion", fem2);

  erosion(fem2, ouv, 1);
  //imshow("ouv erosion", ouv);
  dilatation(ouv, ouv2, 1);
  imshow("last", ouv2);

  imshow("Laplacian", laplacian);
}

void showHistogram(const string& title, Mat& img) {
  int bins = 256;             // number of bins
  int nc = img.channels();    // number of channels
  vector<Mat> hist(nc);       // array for storing the histograms
  vector<Mat> canvas(nc);     // images for displaying the histogram
  int hmax[3] = {0,0,0};      // peak value for each histogram
  for (unsigned int i = 0; i < hist.size(); i++)
    hist[i] = Mat::zeros(1, bins, CV_32SC1);
  for (int i = 0 ; i < img.rows ; i++) {
    for (int j = 0 ; j < img.cols ; j++) {
      for (int k = 0 ; k < nc ; k++) {
        uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
        hist[k].at<int>(val) += 1;
      }
    }
  }
  for (int i = 0 ; i < nc ; i++) {
    for (int j = 0 ; j < bins-1 ; j++)
      hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
  }
  const char* wname[3] = { "blue", "green", "red" };
  Scalar colors[3] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };

  for (int i = 0 ; i < nc ; i++) {
    canvas[i] = Mat::ones(125, bins, CV_8UC3);
    for (int j = 0, rows = canvas[i].rows ; j < bins-1 ; j++)
      line(canvas[i], Point(j, rows), Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])), nc == 1 ? Scalar(200,200,200) : colors[i], 1, 8, 0);
    string t;
    t.append(title);
    if (nc == 1)
      t.append(" value");
    else {
      t.append(" ");
      t.append(wname[i]);
    }
    imshow(t.c_str(), canvas[i]);
  }
}

void drawFingerTips(Mat& destination, vector<Contour*>& contours, Mat& handRegion, int J) {
  int K = 60;
  float theta = 90;

  qds::LinkedList<qm::Vec2i> fingertips;
  for (vector<Contour*>::iterator i = contours.begin() ; i != contours.end() ; i++) {
    fingertips.clear();
    (*i)->findFingertips(fingertips, K, theta, handRegion, true, J, destination);
  }
}

void associateColor(Mat& A, Mat& B, Mat& destination) {
  A.copyTo(destination);
  uchar* bData = (uchar*) B.data;
  uchar* outData = (uchar*) destination.data;
  for (int i = 0 ; i < A.rows ; i++) {
    for (int j = 0 ; j < A.cols ; j++) {
      uchar r = bData[i*B.step + 3*j + 2];
      uchar g = bData[i*B.step + 3*j + 1];
      uchar b = bData[i*B.step + 3*j + 0];
      if (r != 0)
        outData[i*destination.step + 3*j + 2] = r;
      if (g != 0)
        outData[i*destination.step + 3*j + 1] = g;
      if (b != 0)
        outData[i*destination.step + 3*j + 0] = b;
    }
  }
}

void onFingersTrackbar(int, void*) {
  Mat fingers;
  extractedContour.copyTo(fingers);
  Mat imageAndContour;

  associateColor(frame, extractedContour, imageAndContour);
  drawFingerTips(imageAndContour, frameContours, frameFilteredSegmentation, fingersSlider);
  imshow("Fingers", imageAndContour);
}

void multiply(Mat& A, Mat& B, Mat& destination) {
  destination = Mat::zeros(A.rows, A.cols, A.type());
  for (int i = 0 ; i < A.rows ; i++) {
    uchar* rowA = A.ptr<uchar>(i);
    uchar* rowB = B.ptr<uchar>(i);
    uchar* rowDest = destination.ptr<uchar>(i);
    for (int j = 0 ; j < A.cols ; j++) {
      uchar a = rowA[j];
      uchar b = rowB[j];
      if (a < 10 || b < 30)
        rowDest[j] = (uchar) 0;
      else
        rowDest[j] = a;
    }
  }
}


int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "ERROR: please give the config file as a command line argument." << endl;
    exit(1);
  }

  string configFile(argv[1]);
  if (!initFromConfigFile(configFile))
    exit(1);


  cv::initModule_nonfree();

  handImage = imread(IMAGES_FOLDER + "hand1.jpg", CV_LOAD_IMAGE_COLOR);
  Size handImageSize = handImage.size();
  int width = handImageSize.width;
  int height = handImageSize.height;

  cout << "Image size: " << width << "x" << height << endl;

  cvtColor(handImage, greyHand, CV_BGR2GRAY);
  //imshow("Hand", handImage);
  //imshow("Grey hand", greyHand);

  Segmentation handSegmentation(0.9f);
  handSegmentation.loadColorDistributions(DATA_FOLDER + "skin.gmm", DATA_FOLDER + "nonSkin.gmm", DATA_FOLDER + "skin.dis", DATA_FOLDER + "nonSkin.dis");

  Mat handRegion;

  //
  // Segmentation tests
  //

  /*handImage = imread(IMAGES_FOLDER + "hand3.jpg");
  handSegmentation.computeHandRegion(handImage, handRegion, true);
  imshow("hand 3 region", handRegion);
  handSegmentation.learnYCrCbColor(true);

  handImage = imread(IMAGES_FOLDER + "hand1.jpg");
  handSegmentation.computeHandRegion(handImage, handRegion, true);
  imshow("hand 1 region", handRegion);
  handSegmentation.learnYCrCbColor(true);

  handImage = imread(IMAGES_FOLDER + "hand2.jpg");
  handSegmentation.computeHandRegion(handImage, handRegion, true);
  imshow("hand 2 region", handRegion);
  handSegmentation.learnYCrCbColor(true);

  handImage = imread(IMAGES_FOLDER + "hand5.jpg");
  handSegmentation.computeHandRegion(handImage, handRegion, true);
  imshow("hand 5 region with learning", handRegion);
  imwrite("hand5_with_learning.png", handRegion);
  handSegmentation.learnYCrCbColor(true);

  handImage = imread(IMAGES_FOLDER + "hand6.jpg");
  handSegmentation.computeHandRegion(handImage, handRegion, true);
  imshow("hand 6 region", handRegion);
  handSegmentation.learnYCrCbColor(true);

  handImage = imread(IMAGES_FOLDER + "hand7.jpg");
  handSegmentation.computeHandRegion(handImage, handRegion, true);
  imshow("hand 7 region", handRegion);
  handSegmentation.learnYCrCbColor(true);*/


  //
  // Contour extraction tests
  //
  ContourDetector contourDetector;

  if (false) {
    handSegmentation.learnYCrCbColor(false);
    frame = imread(IMAGES_FOLDER + "frame147.png"); // 37
    imshow("hand", frame);
    handSegmentation.computeHandRegion(frame, handRegion);
    cout << handRegion.cols << " ok" <<endl;
    imshow("hand region", handRegion);
    //handSegmentation.learnYCrCbColor(true);

    handRegion.copyTo(frameFilteredSegmentation);

    medianBlur(frameFilteredSegmentation, frameFilteredSegmentation, 11);
    imshow("median", frameFilteredSegmentation);

    //openingB(frameFilteredSegmentation, frameFilteredSegmentation, 3);
    //imshow("filtered", frameFilteredSegmentation);

    /*closingB(frameFilteredSegmentation, frameFilteredSegmentation, 3);
    imshow("filtered", frameFilteredSegmentation);*/

    /*openingB(frameFilteredSegmentation, frameFilteredSegmentation, 5);
    imshow("filtered O", frameFilteredSegmentation);*/

    Mat distT = Mat::zeros(handRegion.rows, handRegion.cols, CV_32FC1);
    distanceTransform(frameFilteredSegmentation, distT, CV_DIST_L2, 5);
    normalize(distT, distT, 0.0, 1.0, NORM_MINMAX);
    imshow("Dist transform", distT);
    threshold(distT, distT, 0.05, 1.0, 0);
    imshow("Dist transform 2", distT);
    closingB(distT, distT, 3);
    imshow("Dist transform 3", distT);

    contourDetector.extract(frameFilteredSegmentation, frameContours, 1, 1000, true);
    extractedContour = contourDetector.getExtractedContoursImage();

    Mat fingers;
    extractedContour.copyTo(fingers);
    Mat imageAndContour;

    associateColor(frame, extractedContour, imageAndContour);
    drawFingerTips(imageAndContour, frameContours, frameFilteredSegmentation, 0);

    namedWindow("Fingers", 1);
    imshow("Fingers", imageAndContour);
    createTrackbar("Fingers", "Fingers", &fingersSlider, fingersSliderMax, onFingersTrackbar);
  }

  // Extract performance test
  /*handImage = imread(IMAGES_FOLDER + "hand12.jpg");
  handSegmentation.learnYCrCbColor(false);
  handSegmentation.computeHandRegion(handImage, handRegion);
  handRegion.copyTo(filteredRegion);
  medianBlur(filteredRegion, filteredRegion, 5);
  openingB(filteredRegion, filteredRegion, 3);
  closingB(filteredRegion, filteredRegion, 3);
  vector<Contour*> conts;

  time_t start, end;
  time(&start);
  int tests = 5000;
  for (int k = 0 ; k < tests ; k++) {
    conts.clear();
    contourDetector.extract(filteredRegion, conts, 1, 1000, true);
  }
  time(&end);
  cout << "Extraction mean time: " << (1000.0 * ((double) difftime(end, start)) / tests) << "ms (total " << difftime(end, start) <<"s for " << tests << " tests)" << endl;

  for (int i = 0 ; i < conts.size() ; i++) {
    delete conts[i];
  }*/

  /*time_t start, end;
  Mat mat(1080, 720, CV_64FC3);
  time(&start);
  for (int k = 0 ; k < 10000 ; k++) {
    uchar* data = mat.data;
    for (int i = 0 ; i < mat.cols*mat.rows ; i++) {
      *data++ = 255;
      *data++ = 255;
      *data++ = 255;
    }
  }
  time(&end);
  cout << "Data access: " << difftime(end, start) << endl;
  time(&start);
  for (int k = 0 ; k < 10000 ; k++) {
    uchar* data = mat.data;
    for (int i = 0 ; i < mat.rows ; i++) {
      for (int j = 0 ; j < mat.cols ; j++) {
        data[i*mat.step+3*j] = 255;
        data[i*mat.step+3*j+1] = 255;
        data[i*mat.step+3*j+2] = 255;
      }
    }
  }
  time(&end);
  cout << "Data [] access: " << difftime(end, start) << endl;
  time(&start);
  uchar* p;
  for (int k = 0 ; k < 10000 ; k++) {
    for (int i = 0 ; i < mat.rows ; i++) {
      p = mat.ptr<uchar>(i);
      for (int j = 0 ; j < mat.cols ; j++) {
        p[j] = 255;
        p[j+1] = 255;
        p[j+2] = 255;
      }
    }
  }
  time(&end);
  cout << "Ptr access: " << difftime(end, start) << endl;*/


  //
  // Video tests
  //
  handSegmentation.learnYCrCbColor(false);
  VideoCapture videoCapture;
  //videoCapture.open(VIDEOS_FOLDER + "hand3.avi");
  videoCapture.open(0);
  if (true && videoCapture.isOpened()) {
    cout << "Video capture." << endl;
    time_t start, end;
    Mat frameSegmentation;
    Mat frameAndContours;
    //vector<Contour*> frameContours;
    int frameCounter = 0;

    namedWindow("Fingers", 1);
    createTrackbar("Fingers", "Fingers", &fingersSlider, fingersSliderMax, onFingersTrackbar);
    for (unsigned int i = 0 ; i < frameContours.size() ; i++)
        delete frameContours[i];
    time(&start);
    while (videoCapture.read(frame)) {
      // Segmentation
      handSegmentation.computeHandRegion(frame, frameSegmentation);
      imshow("Segmentation", frameSegmentation);
      // Filtering
      frameSegmentation.copyTo(frameFilteredSegmentation);
      medianBlur(frameFilteredSegmentation, frameFilteredSegmentation, 11);
      imshow("Median", frameFilteredSegmentation);
      //openingB(frameFilteredSegmentation, frameFilteredSegmentation, 3);
      //closingB(frameFilteredSegmentation, frameFilteredSegmentation, 3);
      //imshow("Filtered segmentation", frameFilteredSegmentation);

      Mat distT = Mat::zeros(frame.rows, frame.cols, CV_32FC1);
      distanceTransform(frameFilteredSegmentation, distT, CV_DIST_L2, 5);
      normalize(distT, distT, 0.0, 1.0, NORM_MINMAX);
      //imshow("Dist transform", distT);
      threshold(distT, distT, 0.05, 1.0, 0);
      //imshow("Dist transform 2", distT);

      //imshow("Filtered segmentation", frameFilteredSegmentation);
      // Contour extraction
      for (unsigned int i = 0 ; i < frameContours.size() ; i++)
        delete frameContours[i];
      frameContours.clear();
      contourDetector.extract(frameFilteredSegmentation, frameContours, 1, 1000, true);

      Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
      contourDetector.maskFromContours(frameContours, mask);
      //imshow("Mask", mask);

      extractedContour = contourDetector.getExtractedContoursImage();
      // Fingers detection
      //frame.copyTo(frameAndContours);
      associateColor(frame, extractedContour, frameAndContours);
      drawFingerTips(frameAndContours, frameContours, frameFilteredSegmentation, 0);

      imshow("Fingers", frameAndContours);

      // Save
     /* stringstream out;
      out << OUT_FOLDER << "\\frame" << frameCounter << ".png";
      stringstream out2;
      out2 << OUT_FOLDER << "\\frame" << frameCounter << "-res.png";
      imwrite(out.str(), frame);
      imwrite(out2.str(), frameAndContours);*/

      imshow("Input", frame);
      frameCounter++;
      time(&end);
      double sec = difftime(end, start);
      double fps = frameCounter / sec;
      cout << "FPS: " << fps << endl;

      if (waitKey(1) >= 0) {
        while (waitKey(30) < 0) { }
      }
    }
    for (unsigned int i = 0 ; i < frameContours.size() ; i++)
      delete frameContours[i];

    time(&end);
    double sec = difftime(end, start);
    double fps = frameCounter / sec;
    cout << "Start: " << start << ", end: " << end << ", FPS: " << fps << endl;
  }
  else
    cerr << "Cannot load video" << endl;

  cvWaitKey(0);

  return 0;
}
