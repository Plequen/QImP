#include "gaussianmixturemodel.h"

using namespace qimp;
using namespace std;
using namespace cv;

GaussianMixtureModel::GaussianMixtureModel() {
  mixtures = 0;
  dimensions = 0;
  weights = NULL;
  probability = new float**[256];
  for (int i = 0 ; i < 256 ; i++) {
    probability[i] = new float*[256];
    for (int j = 0 ; j < 256 ; j++)
      probability[i][j] = new float[256];
  }
}

GaussianMixtureModel::GaussianMixtureModel(int mixtures, int dimensions) {
  this->mixtures = 0;
  this->dimensions = dimensions;
  meanMatrices.reserve(mixtures);
  covMatrices.reserve(mixtures);
  covMatricesI.reserve(mixtures);
  weights = new float[mixtures];
  probability = new float**[256];
  for (int i = 0 ; i < 256 ; i++) {
    probability[i] = new float*[256];
    for (int j = 0 ; j < 256 ; j++)
      probability[i][j] = new float[256];
  }
}


GaussianMixtureModel::~GaussianMixtureModel() {
  delete[] weights;
  for (int i = 0 ; i < 256 ; i++) {
    for (int j = 0 ; j < 256 ; j++)
      delete[] probability[i][j];
    delete[] probability[i];
  }
  delete[] probability;
}

bool GaussianMixtureModel::loadFromFile(const std::string& filename) {
  ifstream file(filename.c_str());
  if (!file) {
    cerr << "Could not load the file. " << endl;
    return false;
  }

  int mixturesNumber = 0, dimensionsNumber = 0;
  file >> mixturesNumber >> dimensionsNumber;
  dimensions = dimensionsNumber;

  meanMatrices.reserve(mixturesNumber);
  covMatrices.reserve(mixturesNumber);
  covMatricesI.reserve(mixturesNumber);
  for (int i = 0 ; i < mixturesNumber ; i++) {
    meanMatrices.push_back(Mat());
    covMatrices.push_back(Mat());
    covMatricesI.push_back(Mat());
  }
  weights = new float[mixturesNumber];

  Mat meanMat(dimensions, 1, CV_64FC1);
  Mat covMat = Mat::zeros(dimensions, dimensions, CV_64FC1);
  float weight = 0.f;

  double* meanMatP = meanMat.ptr<double>(0);
  float value = 0.f;
  for (int i = 0 ; i < mixturesNumber ; i++) {
    // Set mean
    for (int j = 0 ; j < dimensions ; j++) {
      file >> value;
      meanMatP[j] = value;
    }
    // Set diagonal covariance
    for (int j = 0 ; j < dimensions ; j++) {
      file >> value;
      covMat.at<double>(j, j) = value;
    }
    // Set weight
    file >> value;
    weight = value;

    addGaussian(meanMat, covMat, weight);
  }
  file.close();

  return true;
}

void GaussianMixtureModel::addGaussian(cv::Mat& meanMat, cv::Mat& covMat, float weight) {
  meanMat.copyTo(meanMatrices[mixtures]);
  covMat.copyTo(covMatrices[mixtures]);
  covMat.copyTo(covMatricesI[mixtures]);
  invert(covMat, covMatricesI[mixtures]);
  weights[mixtures] = weight;
  mixtures++;
}

float GaussianMixtureModel::getProbability(const cv::Mat& sample) {
  double probability = 0.0;
  Mat diffMatrix;
  sample.copyTo(diffMatrix);
  Mat diffMatrixT(1, dimensions, CV_64FC1);
  Mat expoMat(1, 1, CV_64FC1);

  for (int i = 0 ; i < mixtures ; i++) {
    diffMatrix = sample - meanMatrices[i];
    transpose(diffMatrix, diffMatrixT);
    diffMatrix = covMatricesI[i] * diffMatrix;
    expoMat = diffMatrixT * diffMatrix;
    double expo = expoMat.at<double>(0, 0);
    expo *= (-0.5);

    probability += weights[i] * 1.0 / (pow(2*CV_PI, 1.5) * sqrt(determinant(covMatrices[i]))) * exp(expo);
  }
  return probability;
}

void GaussianMixtureModel::makeLookUpTable() {
  Mat sampleMat(3, 1, CV_64FC1);
  for (int r = 0 ; r < 256 ; r++) {
    cout << ".";
    for (int g = 0 ; g < 256 ; g++) {
      for (int b = 0 ; b < 256 ; b++) {
        sampleMat.at<double>(0, 0) = (double) r;
        sampleMat.at<double>(1, 0) = (double) g;
        sampleMat.at<double>(2, 0) = (double) b;
        probability[r][g][b] = getProbability(sampleMat);
      }
    }
  }
}

bool GaussianMixtureModel::saveLookUpTable(const std::string& filename) {
  ofstream binaryFile(filename.c_str(), ios::out | ios::binary);
  if (!binaryFile) {
    cerr << "Could not load the file." << endl;
    return false;
  }
  for (int r = 0 ; r < 256 ; r++) {
    for (int g = 0 ; g < 256 ; g++)
      binaryFile.write(reinterpret_cast<char*> (probability[r][g]), sizeof(float) * 256);
  }
  binaryFile.close();
  return true;
}

bool GaussianMixtureModel::loadLookUpTable(const string& filename) {
  ifstream binaryFile(filename.c_str(), ios::in | ios::binary);
  if (!binaryFile) {
    cerr << "Could not load the file." << endl;
    return false;
  }
  for (int r = 0 ; r < 256 ; r++) {
    for (int g = 0 ; g < 256 ; g++)
      binaryFile.read(reinterpret_cast<char*> (probability[r][g]), sizeof(float) * 256);
  }
  binaryFile.close();
  return true;
}

float GaussianMixtureModel::getProbabilityByLookUp(int r, int g, int b) {
  return probability[r][g][b];
}

void GaussianMixtureModel::printLookUpTable() {
  cout << "-----------------------" << endl;
  cout << "Look up table extract" << endl;
  for (int r = 0 ; r < 10 ; r++) {
    for (int g = 0 ; g < 10 ; g++)
      cout << probability[r][g][0] << ", ";
    cout << endl;
  }
  cout << "-----------------------" << endl;
}
