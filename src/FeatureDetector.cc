#include "FeatureDetector.h"
#include "Matrix.h"
#include <nan.h>
#include <stdio.h>

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >=4

Persistent<FunctionTemplate> FeatureDetector::constructor;

void
FeatureDetector::Init(Handle<Object> target) {
  NanScope();

  // Constructor
  Local<FunctionTemplate> ctor = NanNew<FunctionTemplate>(FeatureDetector::New);
  NanAssignPersistent(constructor, ctor);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(NanNew("FeatureDetector"));

  // Prototype
  NODE_SET_PROTOTYPE_METHOD(ctor, "detect", Detect);
  
  target->Set(NanNew("FeatureDetector"), ctor->GetFunction());
};

NAN_METHOD(FeatureDetector::New) {
  NanScope();

  if (args.This()->InternalFieldCount() == 0){
    JSTHROW_TYPE("Cannot Instantiate without new")
  }

  FeatureDetector* detector;
  if (args.Length() == 1){
    detector = new FeatureDetector(std::string(*NanAsciiString(args[0]->ToString())));
  } else {
    detector = new FeatureDetector("SURF");
  }

  detector->Wrap(args.Holder());
  NanReturnValue(args.Holder());
}

FeatureDetector::FeatureDetector(const std::string& detectorType){
  detector = cv::FeatureDetector::create(std::string(detectorType));
}

NAN_METHOD(FeatureDetector::Detect){
  SETUP_FUNCTION(FeatureDetector)
  Matrix *im = ObjectWrap::Unwrap<Matrix>(args[0]->ToObject());
  std::vector<cv::KeyPoint> keypoints;

  try{
    self->detector->detect(im->mat, keypoints);
    NanReturnUndefined();
  } catch(cv::Exception& e ){
    const char* err_msg = e.what();
    NanThrowError(err_msg);
    NanReturnUndefined();
  }
}

#endif