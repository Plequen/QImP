#include "segmentation.h"
//#include "../contours/contourdetector.h"

using namespace qimp;
using namespace std;
using namespace cv;

Segmentation::Segmentation(float probaBlendFactor) {
  histograms = new HistogramsHistory(5, 16);
  fLearned = false;
  this->probaBlendFactor = probaBlendFactor;
}

Segmentation::~Segmentation() {
  delete histograms;
  histograms = NULL;
}

bool Segmentation::loadColorDistributions(const string& skinFilename, const string& nonSkinFilename, const string& skinLookUpTableFilename, const string& nonSkinLookUpTableFilename) {
  if (!skinColor.loadLookUpTable(skinLookUpTableFilename)) {
    if (!skinColor.loadFromFile(skinFilename)) {
      cerr << "Could not load the skin color distribution." << endl;
      return false;
    }
    cout << "Make a lookup table for the skin color distribution" << endl;
    skinColor.makeLookUpTable();
    cout << " done" << endl;
    if (!skinColor.saveLookUpTable(skinLookUpTableFilename)) {
       cerr << "Could not save the skin color lookup table." << endl;
      return false;
    }
    skinColor.printLookUpTable();
  }
  if (!nonSkinColor.loadLookUpTable(nonSkinLookUpTableFilename)) {
    if (!nonSkinColor.loadFromFile(nonSkinFilename)) {
      cerr << "Could not load the non skin color distribution." << endl;
      return false;
    }
    cout << "Make a lookup table for the non skin color distribution" << endl;
    nonSkinColor.makeLookUpTable();
    cout << " done" << endl;
    if (!nonSkinColor.saveLookUpTable(nonSkinLookUpTableFilename)) {
       cerr << "Could not save the non skin color lookup table." << endl;
      return false;
    }
    nonSkinColor.printLookUpTable();
  }
  return true;
}

bool Segmentation::loadHandMask(const string& filename) {
  handMask = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  flip(handMask, handMask, 0);
  return true;
}

bool Segmentation::initHandMask(const cv::Mat& image) {
  handMask = Mat::zeros(image.rows, image.cols, CV_8UC1);
  return true;
}

void Segmentation::handRegionViaGMM(const Mat& image, cv::Mat& destination, bool fScreenShot) {
  destination = Mat::zeros(image.rows, image.cols, CV_8UC1);

  // Convert BGR to YCrCb
  cvtColor(image, imageYCrCb, CV_BGR2YCrCb);

  Mat screenShotImage;
  if (fScreenShot)
    screenShotImage = Mat::zeros(image.rows, image.cols, CV_8UC1);

  // Segmentation using color distributions
  int handRegionPixels = 0;
  const uchar* imageRowI;
  uchar* imageYCrCbRowI;
  uchar* handRegionRowI;
  uchar* screenShotRowI;
  uchar r, g, b, y, cr, cb;
  float skinProbability, nonSkinProbability, probability;
  for (int i = 0 ; i < image.rows ; i++) {
    imageRowI = image.ptr<uchar>(i);
    imageYCrCbRowI = imageYCrCb.ptr<uchar>(i);
    handRegionRowI = destination.ptr<uchar>(i);
    screenShotRowI = screenShotImage.ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      r = imageRowI[3*j+2];
      g = imageRowI[3*j+1];
      b = imageRowI[3*j];

      skinProbability = skinColor.getProbabilityByLookUp(r, g, b);
      nonSkinProbability = nonSkinColor.getProbabilityByLookUp(r, g, b);

      if (fLearned) {
        y = imageYCrCbRowI[3*j];
        cr = imageYCrCbRowI[3*j+1];
        cb = imageYCrCbRowI[3*j+2];

        float skinProbabilityViaHistory = histograms->getProbability(y, cr, cb);

        float probaWithoutLearning = skinProbability / nonSkinProbability;

        if (i == 130 && j == 80) {
          //cout << "floor ~" << endl;
          //cout << "skin proba " << skinProbability << endl;
          //cout << "non skin proba " << nonSkinProbability << endl;
          //cout << "skin proba via history " << skinProbabilityViaHistory << endl;
          //cout << "proba without history " << probaWithoutLearning << endl;
          skinProbability = (1 - probaBlendFactor) * skinProbability + probaBlendFactor * skinProbabilityViaHistory;
          //cout << "skin proba blend " << skinProbability << endl;
          //cout << "proba with history " << (skinProbability / nonSkinProbability) << endl;
          //cout << endl;
        }
        else if (i == 400 && j == 190) {
          //cout << "hand ~" << endl;
          //cout << "skin proba " << skinProbability << endl;
          //cout << "non skin proba " << nonSkinProbability << endl;
          //cout << "skin proba via history " << skinProbabilityViaHistory << endl;
          //cout << "proba without history " << probaWithoutLearning << endl;
          skinProbability = (1 - probaBlendFactor) * skinProbability + probaBlendFactor * skinProbabilityViaHistory;
          //cout << "skin proba blend " << skinProbability << endl;
          //cout << "proba with history " << (skinProbability / nonSkinProbability) << endl;
          //cout << endl;
        }
        else {
          skinProbability = (1 - probaBlendFactor) * skinProbability + probaBlendFactor * skinProbabilityViaHistory;
        }

        if (probaWithoutLearning < 0.4 && skinProbability / nonSkinProbability >= 0.4) {
          //cout << "Not skin thanks to learning" << endl;
        }
        if (probaWithoutLearning >= 0.4 && skinProbability / nonSkinProbability < 0.4) {
          //cout << "In skin thanks to learning" << endl;
        }
      }

      probability = skinProbability / nonSkinProbability;
      if (probability >= 0.4) {
        handRegionRowI[j] = 255;
        handRegionPixels++;
      }
      if (fScreenShot) {
        probability = sqrt(sqrt(probability)) * 255;
        if (probability > 255)
          probability = 255;
        screenShotRowI[j] = (uchar) probability;
      }
    }
  }
  if (fScreenShot) {
    imwrite("screenshot_handprob.png", screenShotImage);
  }
 // handMask = destination;
}

void Segmentation::handRegionViaGMM2(const Mat& image, Mat& destination, bool fScreenShot) {
  destination = Mat::zeros(image.rows, image.cols, CV_8UC1);

  // Convert BGR to YCrCb
  cvtColor(image, imageYCrCb, CV_BGR2YCrCb);

  /*Mat channelsYCrCb[3];
  split(imageYCrCb, channelsYCrCb);
  //imshow("Y channel", channelsYCrCb[0]);
  //imshow("Cr channel", channelsYCrCb[1]);
  //imshow("Cb channel", channelsYCrCb[2]);

  int quantification = 16;
  int step = 256 / quantification;
  Mat qChannelsYCrCb[3];
  Mat qua;
  channelsYCrCb[1].copyTo(qua);
  split(imageYCrCb, qChannelsYCrCb);
  uchar *rows0, *rows1, *rows2, *rows00, *rows11, *rows22, *rows3;
  for (int i = 0 ; i < image.rows ; i++) {
    rows0 = channelsYCrCb[0].ptr<uchar>(i);
    rows1 = channelsYCrCb[1].ptr<uchar>(i);
    rows2 = channelsYCrCb[2].ptr<uchar>(i);
    rows00 = qChannelsYCrCb[0].ptr<uchar>(i);
    rows11 = qChannelsYCrCb[1].ptr<uchar>(i);
    rows22 = qChannelsYCrCb[2].ptr<uchar>(i);
    rows3 = qua.ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      rows00[j] = (rows0[j] / step) * step;
      rows11[j] = (rows1[j] / step) * step;
      rows22[j] = (rows2[j] / step) * step;
      rows3[j] = (rows1[j] / step) * step;
    }
  }

  //imshow("Y q channel", qChannelsYCrCb[0]);
  imshow("Cr q channel", qChannelsYCrCb[1]);
  imshow("Cb q channel", qChannelsYCrCb[2]);

  ContourDetector detector;
  Mat dest;
  detector.contourImageFromGray(qChannelsYCrCb[1], dest, 1);
  imshow("Cont", dest);

  quantification = 8;
  step = 256 / quantification;
  for (int i = 0 ; i < image.rows ; i++) {
    rows0 = channelsYCrCb[0].ptr<uchar>(i);
    rows1 = channelsYCrCb[1].ptr<uchar>(i);
    rows2 = channelsYCrCb[2].ptr<uchar>(i);
    rows00 = qChannelsYCrCb[0].ptr<uchar>(i);
    rows11 = qChannelsYCrCb[1].ptr<uchar>(i);
    rows22 = qChannelsYCrCb[2].ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      rows00[j] = (rows0[j] / step) * step;
      rows11[j] = (rows1[j] / step) * step;
      rows22[j] = (rows2[j] / step) * step;
    }
  }

  //imshow("Y q2 channel", qChannelsYCrCb[0]);
  imshow("Cr q2 channel", qChannelsYCrCb[1]);
  //imshow("Cb q2 channel", qChannelsYCrCb[2]);


  quantification = 4;
  step = 256 / quantification;
  for (int i = 0 ; i < image.rows ; i++) {
    rows1 = channelsYCrCb[1].ptr<uchar>(i);
    rows11 = qChannelsYCrCb[1].ptr<uchar>(i);
    rows2 = channelsYCrCb[2].ptr<uchar>(i);
    rows22 = qChannelsYCrCb[2].ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      rows11[j] = (rows1[j] / step) * step;
      rows22[j] = (rows2[j] / step) * step;
    }
  }
  imshow("Cr q3 channel", qChannelsYCrCb[1]);
  //imshow("Cb q3 channel", qChannelsYCrCb[2]);

  quantification = 32;
  step = 256 / quantification;
  for (int i = 0 ; i < image.rows ; i++) {
    rows1 = channelsYCrCb[1].ptr<uchar>(i);
    rows11 = qChannelsYCrCb[1].ptr<uchar>(i);
    rows2 = channelsYCrCb[2].ptr<uchar>(i);
    rows22 = qChannelsYCrCb[2].ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      rows11[j] = (rows1[j] / step) * step;
      rows22[j] = (rows2[j] / step) * step;
    }
  }
  imshow("Cr q4 channel", qChannelsYCrCb[1]);
  //imshow("Cb q4 channel", qChannelsYCrCb[2]);

  quantification = 64;
  step = 256 / quantification;
  for (int i = 0 ; i < image.rows ; i++) {
    rows1 = channelsYCrCb[1].ptr<uchar>(i);
    rows11 = qChannelsYCrCb[1].ptr<uchar>(i);
    rows2 = channelsYCrCb[2].ptr<uchar>(i);
    rows22 = qChannelsYCrCb[2].ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      rows11[j] = (rows1[j] / step) * step;
      rows22[j] = (rows2[j] / step) * step;
    }
  }
  imshow("Cr q5 channel", qChannelsYCrCb[1]);
  //imshow("Cb q5 channel", qChannelsYCrCb[2]);


  threshold(channelsYCrCb[1], qChannelsYCrCb[1], 126, 255, 0);
  imshow("Cr q6 channel", qChannelsYCrCb[1]);
  //imshow("Cb q6 channel", qChannelsYCrCb[2]);


  int size = 3;
  Mat element = getStructuringElement(MORPH_RECT, Size( 2*size + 1, 2*size+1 ),
                                       Point( size, size ) );

  /// Apply the dilatation operation
  //medianBlur(qChannelsYCrCb[1], qChannelsYCrCb[1], 3);
  erode( qChannelsYCrCb[1], qChannelsYCrCb[1], element );
  dilate( qChannelsYCrCb[1], qChannelsYCrCb[1], element );
  imshow("eroded", qChannelsYCrCb[1]);

  Mat seg;
  qua.copyTo(seg);
  for (int i = 0 ; i < image.rows ; i++) {
    rows3 = qua.ptr<uchar>(i);
    rows1 = qChannelsYCrCb[1].ptr<uchar>(i);
    rows2 = seg.ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      if (rows1[j] == 0)
        rows2[j] = 0;
    }
  }
  imshow("seg", seg);

  Mat imageHSV;
  cvtColor(image, imageHSV, CV_BGR2HSV);
  Mat channelsHSV[3];
  split(imageHSV, channelsHSV);
  imshow("H channel", channelsHSV[0]);
  imshow("S channel", channelsHSV[1]);
  imshow("V channel", channelsHSV[2]);


  string name = "image YrCrCb ";
  if (n == 0)
    name.append("0");
  if (n == 1)
    name.append("1");
  if (n == 2)
    name.append("2");
  imshow(name, imageYCrCb);

  n++;*/

  Mat screenShotImage;
  if (fScreenShot)
    screenShotImage = Mat::zeros(image.rows, image.cols, CV_8UC1);

  // Segmentation using color distributions
  int handRegionPixels = 0;
  const uchar* imageRowI;
  uchar* imageYCrCbRowI;
  uchar* handRegionRowI;
  uchar* screenShotRowI;
  uchar r, g, b, y, cr, cb;
  float skinProbability, nonSkinProbability, probability;
  for (int i = 0 ; i < image.rows ; i++) {
    imageRowI = image.ptr<uchar>(i);
    imageYCrCbRowI = imageYCrCb.ptr<uchar>(i);
    handRegionRowI = destination.ptr<uchar>(i);
    screenShotRowI = screenShotImage.ptr<uchar>(i);
    for (int j = 0 ; j < image.cols ; j++) {
      r = imageRowI[3*j+2];
      g = imageRowI[3*j+1];
      b = imageRowI[3*j];

      skinProbability = skinColor.getProbabilityByLookUp(r, g, b);
      nonSkinProbability = nonSkinColor.getProbabilityByLookUp(r, g, b);

      if (fLearned) {
        y = imageYCrCbRowI[3*j];
        cr = imageYCrCbRowI[3*j+1];
        cb = imageYCrCbRowI[3*j+2];

        float skinProbabilityViaHistory = histograms->getProbability(y, cr, cb);

        float probaWithoutLearning = skinProbability / nonSkinProbability;

        if (i == 130 && j == 80) {
          //cout << "floor ~" << endl;
          //cout << "skin proba " << skinProbability << endl;
          //cout << "non skin proba " << nonSkinProbability << endl;
          //cout << "skin proba via history " << skinProbabilityViaHistory << endl;
          //cout << "proba without history " << probaWithoutLearning << endl;
          skinProbability = (1 - probaBlendFactor) * skinProbability + probaBlendFactor * skinProbabilityViaHistory;
          //cout << "skin proba blend " << skinProbability << endl;
          //cout << "proba with history " << (skinProbability / nonSkinProbability) << endl;
          //cout << endl;
        }
        else if (i == 400 && j == 190) {
          //cout << "hand ~" << endl;
          //cout << "skin proba " << skinProbability << endl;
          //cout << "non skin proba " << nonSkinProbability << endl;
          //cout << "skin proba via history " << skinProbabilityViaHistory << endl;
          //cout << "proba without history " << probaWithoutLearning << endl;
          skinProbability = (1 - probaBlendFactor) * skinProbability + probaBlendFactor * skinProbabilityViaHistory;
          //cout << "skin proba blend " << skinProbability << endl;
          //cout << "proba with history " << (skinProbability / nonSkinProbability) << endl;
          //cout << endl;
        }
        else {
          skinProbability = (1 - probaBlendFactor) * skinProbability + probaBlendFactor * skinProbabilityViaHistory;
        }

        if (probaWithoutLearning < 0.4 && skinProbability / nonSkinProbability >= 0.4) {
          //cout << "Not skin thanks to learning" << endl;
        }
        if (probaWithoutLearning >= 0.4 && skinProbability / nonSkinProbability < 0.4) {
          //cout << "In skin thanks to learning" << endl;
        }
      }

      probability = skinProbability / nonSkinProbability;
      if (probability >= 0.4) {
        handRegionRowI[j] = 255;
        handRegionPixels++;
      }
      if (fScreenShot) {
        probability = sqrt(sqrt(probability)) * 255;
        if (probability > 255)
          probability = 255;
        screenShotRowI[j] = (uchar) probability;
      }
    }
  }
  if (fScreenShot) {
    imwrite("C:\\Users\\Quentin\\Documents\\Workspace\\HandTracking\\screenshot_handprob.png", screenShotImage);
  }
 // handMask = destination;
}


void Segmentation::computeHandRegion(const Mat& image, Mat& destination, bool fScreenShot) {
  handRegionViaGMM(image, destination, fScreenShot);
}

void Segmentation::learnYCrCbColor(bool learn) {
  if (!learn) {
    fLearned = false;
    histograms->reset();
    return;
  }
  histograms->learnFromFrame(imageYCrCb, handMask);
  fLearned = true;
}

 void Segmentation::learnYCrCbColor(bool learn, cv::Mat& mask) {
  if (!learn) {
    fLearned = false;
    histograms->reset();
    return;
  }
  histograms->learnFromFrame(imageYCrCb, mask);
  fLearned = true;
}

