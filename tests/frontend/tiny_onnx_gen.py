
import json, sys
sys.path.insert(0, 'src')

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

W = np.random.randn(8, 3, 3, 3).astype(np.float32)
b = np.random.randn(8).astype(np.float32)

graph = helper.make_graph(
    nodes=[
        helper.make_node('Conv',  ['data','w','b'], ['conv_out'],
                        kernel_shape=[3,3], pads=[1,1,1,1]),
        helper.make_node('Relu',  ['conv_out'], ['output']),
    ],
    name='tiny',
    inputs=[helper.make_tensor_value_info('data', TensorProto.FLOAT, [1,3,32,32])],
    outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,8,32,32])],
    initializer=[
        numpy_helper.from_array(W, name='w'),
        numpy_helper.from_array(b, name='b'),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
onnx.save(model, 'tiny_smoke.onnx')