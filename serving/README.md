# Just an attempt to build a light-weight TensorFlow-Serving alternative in Go

Prepare the model by running `training/create.py`.

Now this model can be served either by `tensorflow-serving` using

```console
server@host $ cd server/tensorflow-serving
server@host $ ./run.sh
```

or

```console
server@host $ cd server/tensorflow-serving-lite
server@host $ go build tensorflow-serving-lite.go && ./tensorflow-serving-lite
```

TensorFlow-Serving-Lite has currently hard-coded end-points.