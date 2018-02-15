// 2018, Patrick Wieschollek <mail@patwie.com>
package main

import (
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io/ioutil"
)

func main() {

	// generate new empty graph
	graph := tf.NewGraph()

	// create new session
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}

	// import graph structure from saved model
	model, err := ioutil.ReadFile("./exported/graph.pb")
	if err != nil {
		panic(err)
	}
	err = graph.Import(model, "")
	if err != nil {
		panic(err)
	}

	// initialize all values
	initOp := graph.Operation("init")
	_, err = sess.Run(nil, nil, []*tf.Operation{initOp})
	if err != nil {
		panic(err)
	}

	// restore weight values
	checkpointPathOp := graph.Operation("save/Const").Output(0)
	checkpointRestoreOp := graph.Operation("save/restore_all")
	checkpointPathTensor, err := tf.NewTensor("./exported/my_model")
	if err != nil {
		panic(err)
	}
	_, err = sess.Run(map[tf.Output]*tf.Tensor{checkpointPathOp: checkpointPathTensor}, nil, []*tf.Operation{checkpointRestoreOp})

	// run inference
	inputOp := graph.Operation("input").Output(0)
	outputOp := graph.Operation("output").Output(0)
	inputData, err := tf.NewTensor([][]float32{{1, 1}})
	if err != nil {
		panic(err)
	}

	denseWOp := graph.Operation("dense/kernel").Output(0)
	denseBOp := graph.Operation("dense/bias").Output(0)
	output, err := sess.Run(map[tf.Output]*tf.Tensor{inputOp: inputData}, []tf.Output{outputOp, denseWOp, denseBOp}, nil)
	fmt.Printf("input           %v\n", inputData.Value())
	fmt.Printf("output          %v\n", output[0].Value())
	fmt.Printf("dense/kernel:0  %v\n", output[1].Value())
	fmt.Printf("dense/bias:0    %v\n", output[2].Value())

}
