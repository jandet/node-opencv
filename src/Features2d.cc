#include "Features2d.h"
#include "Matrix.h"
#include "Calib3d.h"
#include <nan.h>
#include <stdio.h>

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >=4

inline Local<Object> matrixFromMat(cv::Mat &input)
{
  Local<Object> matrixWrap = NanNew(Matrix::constructor)->GetFunction()->NewInstance();
  Matrix *matrix = ObjectWrap::Unwrap<Matrix>(matrixWrap);
  matrix->mat = input;

  return matrixWrap;
}

void
Features::Init(Handle<Object> target) {
  NanScope();

  NODE_SET_METHOD(target, "ImageSimilarity", Similarity);
  NODE_SET_METHOD(target, "FindKnownObject", FindKnownObject);
};

class AsyncDetectSimilarity : public NanAsyncWorker {
 public:
  AsyncDetectSimilarity(NanCallback *callback, cv::Mat image1, cv::Mat image2) : NanAsyncWorker(callback), image1(image1), image2(image2), dissimilarity(0) {}
  ~AsyncDetectSimilarity() {}

  void Execute () {

      cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
      cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("ORB");
      cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

      std::vector<cv::DMatch> matches;

      cv::Mat descriptors1 = cv::Mat();
      cv::Mat descriptors2 = cv::Mat();

      std::vector<cv::KeyPoint> keypoints1;
      std::vector<cv::KeyPoint> keypoints2;

      detector->detect(image1, keypoints1);
      detector->detect(image2, keypoints2);

      extractor->compute(image1, keypoints1, descriptors1);
      extractor->compute(image2, keypoints2, descriptors2);

      matcher->match(descriptors1, descriptors2, matches);

      double max_dist = 0;
      double min_dist = 100;

      //-- Quick calculation of max and min distances between keypoints
      for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
      //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
      //-- small)
      //-- PS.- radiusMatch can also be used here.
      std::vector<cv::DMatch> good_matches;
      double good_matches_sum = 0.0;

      for (int i = 0; i < descriptors1.rows; i++ ) {
        double distance = matches[i].distance;
        if (distance <= std::max(2*min_dist, 0.02)) {
          good_matches.push_back(matches[i]);
          good_matches_sum += distance;
        }
      }

      dissimilarity = (double)good_matches_sum / (double)good_matches.size();

  }

  void HandleOKCallback () {
    NanScope();

    Handle<Value> argv[2];

    argv[0] = NanNull();
    argv[1] = NanNew<Number>(dissimilarity);

    callback->Call(2, argv);

  }

  private:
    cv::Mat image1;
    cv::Mat image2;
    double dissimilarity;

};


NAN_METHOD(Features::Similarity) {
  NanScope();

  REQ_FUN_ARG(2, cb);

  cv::Mat image1 = ObjectWrap::Unwrap<Matrix>(args[0]->ToObject())->mat;
  cv::Mat image2 = ObjectWrap::Unwrap<Matrix>(args[1]->ToObject())->mat;

  NanCallback *callback = new NanCallback(cb.As<Function>());

  NanAsyncQueueWorker( new AsyncDetectSimilarity(callback, image1, image2) );
  NanReturnUndefined();

};

class AsyncFindKnownObject : public NanAsyncWorker {
 public:
  AsyncFindKnownObject(NanCallback *callback, cv::Mat image_obj, cv::Mat image_scene) : NanAsyncWorker(callback), image_obj(image_obj), image_scene(image_scene) {}
  ~AsyncFindKnownObject() {}

  void Execute () {

      cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SURF");
      cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SURF");
      cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

      std::vector<cv::DMatch> matches;

      cv::Mat descriptors_obj = cv::Mat();
      cv::Mat descriptors_scene = cv::Mat();

      std::vector<cv::KeyPoint> keypoints_obj;
      std::vector<cv::KeyPoint> keypoints_scene;

      detector->detect(image_obj, keypoints_obj);
      detector->detect(image_scene, keypoints_scene);

      extractor->compute(image_obj, keypoints_obj, descriptors_obj);
      extractor->compute(image_scene, keypoints_scene, descriptors_scene);

      matcher->match(descriptors_obj, descriptors_scene, matches);

      double max_dist = 0;
      double min_dist = 100;

      //-- Quick calculation of max and min distances between keypoints
      for (int i = 0; i < descriptors_obj.rows; i++) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
      //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
      //-- small)
      //-- PS.- radiusMatch can also be used here.
      std::vector<cv::DMatch> good_matches;

      for (int i = 0; i < descriptors_obj.rows; i++ ) {
        if (matches[i].distance <= std::max(2*min_dist, 0.02)) {
          good_matches.push_back(matches[i]);
        }
      }

      drawMatches( image_obj, keypoints_obj, image_scene, keypoints_scene,
                   good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

      //-- Localize the object
      std::vector<cv::Point2f> obj;
      std::vector<cv::Point2f> scene;

      for( int i = 0; i < good_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_obj[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
      }

      cv::Mat H = findHomography( obj, scene, CV_RANSAC );

      //-- Get the corners from the image_1 ( the object to be "detected" )
      std::vector<cv::Point2f> obj_corners(4);
      obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image_obj.cols, 0 );
      obj_corners[2] = cvPoint( image_obj.cols, image_obj.rows ); obj_corners[3] = cvPoint( 0, image_obj.rows );
      std::vector<cv::Point2f> scene_corners(4);

      perspectiveTransform( obj_corners, scene_corners, H);

      //-- Draw lines between the corners (the mapped object in the scene - image_2 )
      line( img_matches, scene_corners[0] + cv::Point2f( image_obj.cols, 0), scene_corners[1] + cv::Point2f( image_obj.cols, 0), cv::Scalar(0, 255, 0), 4 );
      line( img_matches, scene_corners[1] + cv::Point2f( image_obj.cols, 0), scene_corners[2] + cv::Point2f( image_obj.cols, 0), cv::Scalar( 0, 255, 0), 4 );
      line( img_matches, scene_corners[2] + cv::Point2f( image_obj.cols, 0), scene_corners[3] + cv::Point2f( image_obj.cols, 0), cv::Scalar( 0, 255, 0), 4 );
      line( img_matches, scene_corners[3] + cv::Point2f( image_obj.cols, 0), scene_corners[0] + cv::Point2f( image_obj.cols, 0), cv::Scalar( 0, 255, 0), 4 );

  }

  void HandleOKCallback () {
    NanScope();

    Handle<Value> argv[2];

    argv[0] = NanNull();

    Local<Object> matches = matrixFromMat(img_matches);
    argv[1] = matches;

    callback->Call(2, argv);

  }

  private:
    cv::Mat image_obj;
    cv::Mat image_scene;
    cv::Mat img_matches;

};

NAN_METHOD(Features::FindKnownObject) {
  NanScope();

  REQ_FUN_ARG(2, cb);

  cv::Mat image_obj = ObjectWrap::Unwrap<Matrix>(args[0]->ToObject())->mat;
  cv::Mat image_scene = ObjectWrap::Unwrap<Matrix>(args[1]->ToObject())->mat;

  NanCallback *callback = new NanCallback(cb.As<Function>());

  NanAsyncQueueWorker( new AsyncFindKnownObject(callback, image_obj, image_scene) );
  NanReturnUndefined();

};


#endif
