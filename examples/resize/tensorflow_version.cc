// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
Loads, resizes and saves an image using TensorFlow only.
*/

int main() {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  std::string fn = "Grace_Hopper.png";

  auto net1 = tensorflow::ops::ReadFile(root, fn);
  auto net2 = tensorflow::ops::DecodePng(root, net1);
  auto net3 = tensorflow::ops::Cast(root, net2, tensorflow::DT_FLOAT);
  auto net4 = tensorflow::ops::ExpandDims(root, net3, 0);
  auto net5 = tensorflow::ops::ResizeBilinear(root, net4, tensorflow::ops::Const(root, {2 * 606, 2 * 517}));
  auto net6 = tensorflow::ops::Reshape(root, net5, tensorflow::ops::Const(root, {2 * 606, 2 * 517, 3}));
  auto net7 = tensorflow::ops::Cast(root, net6, tensorflow::DT_UINT8);
  auto net8 = tensorflow::ops::EncodeJpeg(root, net7);

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::ClientSession session(root);

  // Run and fetch v
  TF_CHECK_OK(session.Run({net8}, &outputs));
  std::ofstream("output.jpg", std::ios::binary) << outputs[0].scalar<std::string>()();

  return 0;
}
