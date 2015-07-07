// Template class for KeyPoints

#include "OpenCV.h"

class KeyPoint: public node::ObjectWrap {
  public:
    Point2f pt;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;

    static Persistent<FunctionTemplate> constructor;
    static void Init(Handle<Object> target);
    static NAN_METHOD(New);
    KeyPoint();
    KeyPoint(Point2f _pt, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1);
    KeyPoint(float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1);
};

