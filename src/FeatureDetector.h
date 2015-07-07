#include "OpenCV.h"

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >=4

class FeatureDetector: public node::ObjectWrap {
  private:
    cv::Ptr<cv::FeatureDetector> detector;

  public:
    static Persistent<FunctionTemplate> constructor;
    static void Init(Handle<Object> target);
    static NAN_METHOD(New);

    FeatureDetector(const std::string& type);

    JSFUNC(Detect);

};

#endif
