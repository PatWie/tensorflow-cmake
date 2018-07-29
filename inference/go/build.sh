TF_PATH=`python -c "import tensorflow as tf; print(tf.__path__[0])"`
export LD_LIBRARY_PATH=${TF_PATH}:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${TF_PATH}:${LIBRARY_PATH}

go build inference_go.go

echo "you need to set the following paths:"
echo "export LD_LIBRARY_PATH=${TENSORFLOW_BUILD_DIR}/:\${LD_LIBRARY_PATH}"
echo "export LD_LIBRARY_PATH=${TF_PATH}/:\${LD_LIBRARY_PATH}"
