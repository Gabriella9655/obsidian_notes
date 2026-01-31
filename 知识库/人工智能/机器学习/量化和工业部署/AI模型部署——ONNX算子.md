要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证以下三个环节都不出错：

- 算子在 PyTorch 中有实现
- 有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
- ONNX 有相应的算子
对于这三个环节，我们也分别都有以下的添加支持的方法：

- PyTorch 算子
    - 组合现有算子
    - 添加 TorchScript 算子
    - 添加普通 C++ 拓展算子
- 映射方法
    - 为 ATen 算子添加符号函数
    - 为 TorchScript 算子添加符号函数
    - 封装成 torch.autograd.Function 并添加符号函数
- ONNX 算子
    - 使用现有 ONNX 算子
    - 定义新 ONNX 算子

# 为 ATen 算子添加符号函数

>[ATen](https://pytorch.org/cppdocs/#aten) 是 PyTorch 内置的 C++ 张量计算库，PyTorch 算子在底层绝大多数计算都是用 ATen 实现的。

## 获取ATen中算子接口定义

`torch/_C/_VariableFunctions.pyi` 和 `torch/nn/functional.pyi` 是编译PyTorch时本地自动生成的文件，里面包含了ATen算子的Pytorch调用。
通过搜索，我们可以知道 `asinh` 在文件 `torch/_C/_VariableFunctions.pyi` 中，其接口定义为:

```python
def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
```

经过这些步骤，我们确认了缺失的算子名为 `asinh`，它是一个有实现的 ATen 算子。我们还记下了 `asinh` 的调用接口。

## 添加符号函数

符号函数，可以看成是 PyTorch 算子类的一个静态方法。在把 PyTorch 模型转换成 ONNX 模型时，各个 PyTorch 算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换。符号函数的定义一般如下：

```python
def symbolic(g: torch._C.Graph, input_0: torch._C.Value, input_1: torch._C.Value, ...):
```

其中，`torch._C.Graph` 和 `torch._C.Value` 都对应 PyTorch 的 C++ 实现里的一些类。我们在这篇文章不深究它们的细节，只需要知道第一个参数就固定叫 `g`，它表示和计算图相关的内容；后面的每个参数都表示算子的输入，需要和算子的前向推理接口的输入相同。对于 ATen 算子来说，它们的前向推理接口就是上述两个 `.pyi` 文件里的函数接口。

`g` 有一个方法 `op`。在把 PyTorch 算子转换成 ONNX 算子时，需要在符号函数中调用此方法来为最终的计算图添加一个 ONNX 算子。其定义如下：

```python
def op(name: str, input_0: torch._C.Value, input_1: torch._C.Value, ...)
```

其中，第一个参数是算子名称。如果该算子是普通的 ONNX 算子，只需要把它在 ONNX 官方文档里的名称填进去即可（我们稍后再讲其他情况）。

在最简单的情况下，我们只要把 PyTorch 算子的输入用`g.op()`一一对应到 ONNX 算子上即可，并把`g.op()`的返回值作为符号函数的返回值。
```python
from torch.onnx.symbolic_registry import register_op

def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

register_op('asinh', asinh_symbolic, '', 9)
```
这里的`asinh_symbolic`就是`asinh`的符号函数。从除`g`以外的第二个输入参数开始，其输入参数应该严格对应它在 ATen 中的定义：

```python
def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
```

在符号函数的函数体中，`g.op("Asinh", input)`则完成了 ONNX 算子的定义。其中，第一个参数`"Asinh"`是算子在 ONNX 中的名称。至于第二个参数 `input`，如我们刚刚在文档里所见，这个算子只有一个输入，因此我们只要把符号函数的输入参数 `input` 对应过去就行。ONNX 的 `Asinh` 的输出和 ATen 的 `asinh` 的输出是一致的，因此我们直接把 `g.op()` 的结果返回即可。

定义完符号函数后，我们要把这个符号函数和原来的 ATen 算子“绑定”起来。这里，我们要用到 `register_op` 这个 PyTorch API 来完成绑定。如示例所示，只需要一行简单的代码即可把符号函数 `asinh_symbolic` 绑定到算子 `asinh` 上：

```python
register_op('asinh', asinh_symbolic, '', 9)
```

`register_op`的第一个参数是目标 ATen 算子名，第二个是要注册的符号函数，这两个参数很好理解。第三个参数是算子的“域”，对于普通 ONNX 算子，直接填空字符串即可。第四个参数表示向哪个算子集版本注册。我们遵照 ONNX 标准，向第 9 号算子集注册。值得注意的是，这里向第 9 号算子集注册，不代表较新的算子集（第 10 号、第 11 号……）都得到了注册。在示例中，我们先只向第 9 号算子集注册。

整理一下，我们最终的代码如下：

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

from torch.onnx.symbolic_registry import register_op

def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

register_op('asinh', asinh_symbolic, '', 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'asinh.onnx')
```

成功导出的话，`asinh.onnx` 应该长这个样子：

[![](https://user-images.githubusercontent.com/47652064/169744691-f14e4fd4-c777-4562-aaa5-a5bf888f21f8.png)](https://user-images.githubusercontent.com/47652064/169744691-f14e4fd4-c777-4562-aaa5-a5bf888f21f8.png)

### [](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/04_onnx_custom_op.md#%E6%B5%8B%E8%AF%95%E7%AE%97%E5%AD%90)测试算子

在完成了一份自定义算子后，我们一定要测试一下算子的正确性。一般我们要用 PyTorch 运行一遍原算子，再用推理引擎（比如 ONNX Runtime）运行一下 ONNX 算子，最后比对两次的运行结果。对于我们刚刚得到的 `asinh.onnx`，可以用如下代码来验证：

```python
import onnxruntime
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch_output = model(input).detach().numpy()

sess = onnxruntime.InferenceSession('asinh.onnx')
ort_output = sess.run(None, {'0': input.numpy()})[0]

assert np.allclose(torch_output, ort_output)
```

在这份代码里，我们用 PyTorch 做了一遍推理，并把结果转成了 numpy 格式。之后，我们又用 ONNX Runtime 对 onnx 文件做了一次推理。最后，我们使用 `np.allclose` 来保证两个结果张量的误差在一个可以允许的范围内。一切正常的话，运行这段代码后，`assert` 所在行不会报错，程序应该没有任何输出。

## [](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/04_onnx_custom_op.md#%E6%94%AF%E6%8C%81-torchscript-%E7%AE%97%E5%AD%90)