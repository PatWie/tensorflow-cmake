# Use Keras and C++

## Compile (once)

```bash
python configure.py
cmake .
make
```

## Run

```console
user@host python keras_graph.py

...
Using TensorFlow backend.
('input', u'input_plhdr:0')
('output', u'sequential/Output_1/Softmax:0')
('result', array([[9.9999940e-01, 5.8415833e-07]], dtype=float32))
...

user@host ./inference:_cc

...
input           Tensor<type: float shape: [1,2] values: [42 43]>
output          Tensor<type: float shape: [1,2] values: [0.999999404 5.84158329e-07]>
...
```
