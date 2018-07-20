// ComputerGraphics Tuebingen, 2017

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::InferenceContext;

namespace shape_inference {
Status UnchangedShape(InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}
} /* shape_inference */


REGISTER_OP("MatrixAdd")
.Attr("bias: float")
.Attr("T: realnumbertype = DT_FLOAT")
.Input("matrix_a: T")
.Input("matrix_b: T")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  // we require the input to have 4 axes
  ShapeHandle shape_hnd;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &shape_hnd));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &shape_hnd));

  ShapeHandle matrix_a_shape = c->input(0);
  ShapeHandle matrix_b_shape = c->input(1);

  // assert shapes of matrix_a and matrix_b are matching
  TF_RETURN_IF_ERROR(c->Merge(matrix_a_shape, matrix_b_shape, &matrix_a_shape));

  // specify output-shape
  // this could be "c->set_output(0, matrix_a_shape);"
  // but we do it explicitly
  auto B = c->Dim(c->input(0), 0);
  auto M = c->Dim(c->input(0), 1);
  auto N = c->Dim(c->input(0), 2);
  auto D = c->Dim(c->input(0), 3);
  c->set_output(0, c->MakeShape({B, M, N, D}));

  // we can also use the Attr here
  float bias;
  (void) c->GetAttr("bias", &bias);

  return Status::OK();
})
.Doc(R"doc(
Add two matrices and a constant

This computes `A`+`B`+`bias` for two matrices.

matrix_a: A batch of matrices [B, M, N, D].
matrix_b: A batch of matrices [B, M, N, D].
output: A batch of matrices [B, M, N, D] containing the result.
bias: An additional constant term.
)doc");

REGISTER_OP("MatrixAddGrad")
.Attr("bias: float")
.Input("matrix_a: T")
.Input("matrix_b: T")
.Input("gradients: T")
.Output("grad_matrix_a: T")
.Output("grad_matrix_b: T")
.Attr("T: realnumbertype")
.SetShapeFn([](InferenceContext* c) {
  c->set_output(0, c->input(0));
  c->set_output(1, c->input(1));
  return ::tensorflow::Status::OK();
})
.Doc(R"doc(
Returns gradients of "matrix_a + matrix_b + bias".
)doc");


} /* tensorflow */
