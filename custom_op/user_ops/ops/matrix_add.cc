// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace shape_inference {
Status UnchangedShape(InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}
}  // namespace shape_inference

REGISTER_OP("MatrixAdd")
    .Attr("bias: float")
    .Attr("T: realnumbertype = DT_FLOAT")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // we require the input to have 4 axes
      ShapeHandle shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &shape_hnd));

      ShapeHandle x_shape = c->input(0);
      ShapeHandle y_shape = c->input(1);

      // assert shapes of x and y are matching
      TF_RETURN_IF_ERROR(c->Merge(x_shape, y_shape, &x_shape));

      // specify output-shape
      // this could be "c->set_output(0, x_shape);"
      // but we do it explicitly
      auto N = c->Dim(c->input(0), 0);
      auto H = c->Dim(c->input(0), 1);
      auto W = c->Dim(c->input(0), 2);
      auto C = c->Dim(c->input(0), 3);
      c->set_output(0, c->MakeShape({N, H, W, C}));

      // we can also use the Attr here
      float bias;
      (void)c->GetAttr("bias", &bias);

      return Status::OK();
    })
    .Doc(R"doc(
Add two matrices and a constant

This computes `x`+`y`+`bias` for two matrices.

x: A batch of matrices [N, H, W, C].
y: A batch of matrices [N, H, W, C].
bias: An additional constant term.
output: A batch of matrices [N, H, W, C] containing the sum plus bias.
)doc");

REGISTER_OP("MatrixAddGrad")
    .Attr("bias: float")
    .Input("x: T")
    .Input("y: T")
    .Input("gradients: T")
    .Output("grad_a: T")
    .Output("grad_matrix_b: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));  // grad_a has same shape as x
      c->set_output(1, c->input(1));  // grad_b has same shape as y
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Returns gradients of "x + y + bias".
)doc");

}  // namespace tensorflow
