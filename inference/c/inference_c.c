// Patrick Wieschollek <mail@patwie.com>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) { free(data); }

void deallocator(void* ptr, size_t len, void* arg) { free((void*)ptr); }

int main(int argc, char const* argv[]) {
  // load graph
  // ================================================================================
  TF_Buffer* graph_def = read_file("./exported/graph.pb");
  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
    return 1;
  }
  fprintf(stdout, "Successfully imported graph\n");

  // create session
  // ================================================================================
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph, opt, status);
  TF_DeleteSessionOptions(opt);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status));
    return 1;
  }
  fprintf(stdout, "Successfully created session\n");

  // run init operation
  // ================================================================================
  const TF_Operation* init_op = TF_GraphOperationByName(graph, "init");
  const TF_Operation* const* targets_ptr = &init_op;

  TF_SessionRun(sess,
                /* RunOptions */ NULL,
                /* Input tensors */ NULL, NULL, 0,
                /* Output tensors */ NULL, NULL, 0,
                /* Target operations */ targets_ptr, 1,
                /* RunMetadata */ NULL,
                /* Output status */ status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to run init_op: %s\n", TF_Message(status));
    return 1;
  }

  // run restore
  // ================================================================================
  TF_Operation* checkpoint_op = TF_GraphOperationByName(graph, "save/Const");
  TF_Operation* restore_op = TF_GraphOperationByName(graph, "save/restore_all");

  char* checkpoint_path_str = "./exported/my_model";
  size_t checkpoint_path_str_len = strlen(checkpoint_path_str);
  size_t encoded_size = TF_StringEncodedSize(checkpoint_path_str_len);

  // The format for TF_STRING tensors is:
  //   start_offset: array[uint64]
  //   data:         byte[...]
  size_t total_size = sizeof(int64_t) + encoded_size;
  char* input_encoded = (char*)malloc(total_size);
  memset(input_encoded, 0, total_size);
  TF_StringEncode(checkpoint_path_str, checkpoint_path_str_len,
                  input_encoded + sizeof(int64_t), encoded_size, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: something wrong with encoding: %s",
            TF_Message(status));
    return 1;
  }

  TF_Tensor* path_tensor = TF_NewTensor(TF_STRING, NULL, 0, input_encoded,
                                        total_size, &deallocator, 0);

  TF_Output* run_path = (TF_Output*)malloc(1 * sizeof(TF_Output));
  run_path[0].oper = checkpoint_op;
  run_path[0].index = 0;

  TF_Tensor** run_path_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
  run_path_tensors[0] = path_tensor;

  TF_SessionRun(sess,
                /* RunOptions */ NULL,
                /* Input tensors */ run_path, run_path_tensors, 1,
                /* Output tensors */ NULL, NULL, 0,
                /* Target operations */ &restore_op, 1,
                /* RunMetadata */ NULL,
                /* Output status */ status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to run restore_op: %s\n",
            TF_Message(status));
    return 1;
  }

  // gerenate input
  // ================================================================================
  TF_Operation* input_op = TF_GraphOperationByName(graph, "input");
  printf("input_op has %i inputs\n", TF_OperationNumOutputs(input_op));
  float* raw_input_data = (float*)malloc(2 * sizeof(float));
  raw_input_data[0] = 1.f;
  raw_input_data[1] = 1.f;
  int64_t* raw_input_dims = (int64_t*)malloc(2 * sizeof(int64_t));
  raw_input_dims[0] = 1;
  raw_input_dims[1] = 2;

  /*
  TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
      TF_DataType,
      const int64_t* dims, int num_dims,
      void* data, size_t len,
      void (*deallocator)(void* data, size_t len, void* arg),
      void* deallocator_arg);
  */
  // prepare inputs
  TF_Tensor* input_tensor =
      TF_NewTensor(TF_FLOAT, raw_input_dims, 2, raw_input_data,
                   2 * sizeof(float), &deallocator, NULL);

  // void* input_data = TF_TensorData(input_tensor);
  // printf("input_data[0] = %f\n", ((float*)input_data)[0]);
  // printf("input_data[1] = %f\n", ((float*)input_data)[1]);

  TF_Output* run_inputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
  run_inputs[0].oper = input_op;
  run_inputs[0].index = 0;

  TF_Tensor** run_inputs_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
  run_inputs_tensors[0] = input_tensor;

  // prepare outputs
  // ================================================================================
  TF_Operation* output_op = TF_GraphOperationByName(graph, "output");
  // printf("output_op has %i outputs\n", TF_OperationNumOutputs(output_op));

  TF_Output* run_outputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
  run_outputs[0].oper = output_op;
  run_outputs[0].index = 0;

  TF_Tensor** run_output_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
  float* raw_output_data = (float*)malloc(1 * sizeof(float));
  raw_output_data[0] = 1.f;
  int64_t* raw_output_dims = (int64_t*)malloc(1 * sizeof(int64_t));
  raw_output_dims[0] = 1;

  TF_Tensor* output_tensor =
      TF_NewTensor(TF_FLOAT, raw_output_dims, 1, raw_output_data,
                   1 * sizeof(float), &deallocator, NULL);
  run_output_tensors[0] = output_tensor;

  // run network
  // ================================================================================
  TF_SessionRun(sess,
                /* RunOptions */ NULL,
                /* Input tensors */ run_inputs, run_inputs_tensors, 1,
                /* Output tensors */ run_outputs, run_output_tensors, 1,
                /* Target operations */ NULL, 0,
                /* RunMetadata */ NULL,
                /* Output status */ status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(status));
    return 1;
  }

  // printf("output-tensor has %i dims\n", TF_NumDims(run_output_tensors[0]));

  void* output_data = TF_TensorData(run_output_tensors[0]);
  printf("output %f\n", ((float*)output_data)[0]);
  // you do not want see me creating all the other tensors; Enough lines for
  // this simple example!

  // free up stuff
  // ================================================================================
  // I probably missed something here
  TF_CloseSession(sess, status);
  TF_DeleteSession(sess, status);

  TF_DeleteStatus(status);
  TF_DeleteBuffer(graph_def);

  TF_DeleteGraph(graph);

  free((void*)input_encoded);
  free((void*)raw_input_data);
  free((void*)raw_input_dims);
  free((void*)run_inputs);
  return 0;
}

TF_Buffer* read_file(const char* file) {
  FILE* f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  // same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

/*
TF_CAPI_EXPORT extern void TF_SessionRun(
    TF_Session* session,
    // RunOptions
    const TF_Buffer* run_options,
    // Input tensors
    const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
    // Output tensors
    const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // RunMetadata
    TF_Buffer* run_metadata,
    // Output status
    TF_Status*);

*/
