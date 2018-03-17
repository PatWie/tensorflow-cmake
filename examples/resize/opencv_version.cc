// tensorflow/cc/example/example.cc
// Patrick Wieschollek, <mail@patwie.com>

/*
This example loads an image using OpenCV and resizes this image by factor 2 using TensorFlow
before writing it again into a file.
*/

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {

  // read image in OpenCV
  std::string fn = "Grace_Hopper.png";
  cv::Mat image;
  image = cv::imread(fn);
  if (!image.data ) {
    std::cerr <<  "Could not open or find the image" << std::endl ;
    return -1;
  }
  std::cout << "read image " << fn << " with shape " << image.size() << ", "<< image.channels() << std::endl;

  // convert byte to float image
  cv::Mat image_float;
  image.convertTo(image_float, CV_32FC3);
  float *image_float_data = (float*)image_float.data;

  // create input shape
  tensorflow::TensorShape image_shape = tensorflow::TensorShape{1, image.rows, image.cols, image.channels()};
  std::cout << "Input TensorShape ["
            << image_shape.dim_size(0) << ", "
            << image_shape.dim_size(1) << ", "
            << image_shape.dim_size(2) << ", "
            << image_shape.dim_size(3) << "]" << std::endl;

  // create input tensor
  tensorflow::Tensor image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, image_shape);
  // copy data from OpenCv to TensorFlow Tensor
  std::copy_n((char*) image_float_data, image_shape.num_elements() * sizeof(float),
                  const_cast<char*>(image_tensor.tensor_data().data()));

  // create graph for resizing images
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto resized = tensorflow::ops::ResizeBicubic(root, image_tensor, tensorflow::ops::Const(root.WithOpName("size"), {2 * image.rows, 2 * image.cols}));

  // run graph and fetch outputs
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::ClientSession session(root);
  TF_CHECK_OK(session.Run({resized}, &outputs));

  // convert output back to OpenCV matrix
  tensorflow::Tensor output = outputs[0];
  float *result_float_data = output.flat<float>().data();

  std::cout << "Output TensorShape ["
            << output.shape().dim_size(0) << ", "
            << output.shape().dim_size(1) << ", "
            << output.shape().dim_size(2) << ", "
            << output.shape().dim_size(3) << "]" << std::endl;

  cv::Mat resized_image;
  resized_image.create(output.dim_size(1), output.dim_size(2), CV_32FC3);
  std::copy_n((char*) result_float_data , output.shape().num_elements() * sizeof(float), (char*) resized_image.data);

  cv::imwrite("Grace_Hopper_resized.png", resized_image);



  return 0;
}
