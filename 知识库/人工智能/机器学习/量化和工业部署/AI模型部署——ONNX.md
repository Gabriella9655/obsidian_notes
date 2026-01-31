ONNX 是目前模型部署中最重要的中间表示之一。
# pytorch导出ONNX

## torch.onnx.export详解

### 计算图导出方法 (trace vs script)

[TorchScript](https://pytorch.org/docs/stable/jit.html) 是一种序列化和优化 PyTorch 模型的格式，在优化过程中，一个`torch.nn.Module`模型会被转换成 TorchScript 的`torch.jit.ScriptModule`模型。现在， TorchScript 也被常当成一种中间表示使用。我们在[其他文章](https://zhuanlan.zhihu.com/p/486914187)中对 TorchScript 有详细的介绍，这里介绍 TorchScript 仅用于说明 PyTorch 模型转 ONNX的原理。 `torch.onnx.export`中需要的模型实际上是一个`torch.jit.ScriptModule`。而要把普通 PyTorch 模型转一个这样的 TorchScript 模型，有跟踪（trace）和脚本化（script）两种导出计算图的方法。如果给`torch.onnx.export`传入了一个普通 PyTorch 模型（`torch.nn.Module`)，那么这个模型会默认使用跟踪的方法导出。这一过程如下图所示：
![[Pasted image 20240111123920.png]]
```python
import torch

class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x



models = [Model(2), Model(3)]
model_names = ['model_2', 'model_3']

for model, model_name in zip(models, model_names):
    dummy_input = torch.rand(1, 3, 10, 10)
    dummy_output = model(dummy_input)
    model_trace = torch.jit.trace(model, dummy_input)
    model_script = torch.jit.script(model)

    # 跟踪法与直接 torch.onnx.export(model, ...)等价
    torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx', example_outputs=dummy_output)
    # 脚本化必须先调用 torch.jit.sciprt
    torch.onnx.export(model_script, dummy_input, f'{model_name}_script.onnx', example_outputs=dummy_output)
```
在这段代码里，我们定义了一个带循环的模型，模型通过参数`n`来控制输入张量被卷积的次数。之后，我们各创建了一个`n=2`和`n=3`的模型。我们把这两个模型分别用跟踪和脚本化的方法进行导出。 值得一提的是，由于这里的两个模型（`model_trace`, `model_script`)是 TorchScript 模型，`export`函数已经不需要再运行一遍模型了。（如果模型是用跟踪法得到的，那么在执行`torch.jit.trace`的时候就运行过一遍了；而用脚本化导出时，模型不需要实际运行）参数中的`dummy_input`和`dummy_output`仅仅是为了获取输入和输出张量的类型和形状。 

运行上面的代码，我们把得到的4个 onnx 文件用 Netron 可视化:
![[Pasted image 20240111124314.png]]
![[Pasted image 20240111124325.png]]
首先看跟踪法得到的 ONNX 模型结构。可以看出来，对于不同的 `n`,ONNX 模型的结构是不一样的。而用脚本化的话，最终的 ONNX 模型用 `Loop` 节点来表示循环。这样哪怕对于不同的 `n`，ONNX 模型也有同样的结构。 由于推理引擎对静态图的支持更好，通常我们在模型部署时不需要显式地把 PyTorch 模型转成 TorchScript 模型，直接把 PyTorch 模型用 `torch.onnx.export` 跟踪导出即可。了解这部分的知识主要是为了在模型转换报错时能够更好地定位问题是否发生在 PyTorch 转 TorchScript 阶段。
### 参数详解

`torch.onnx.export` 在 `torch.onnx.__init__.py`文件中的定义如下：
```python
def export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None, _retain_param_name=True,
           do_constant_folding=True, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
           enable_onnx_checker=True, use_external_data_format=False):
```
前三个必选参数为模型、模型输入、导出的 onnx 文件名。
#### export_params

模型中是否存储模型权重。一般中间表示包含两大类信息：模型结构和模型权重，这两类信息可以在同一个文件里存储，也可以分文件存储。ONNX 是用同一个文件表示记录模型的结构和权重的。 我们部署时一般都默认这个参数为 True。如果 onnx 文件是用来在不同框架间传递模型（比如 PyTorch 到 Tensorflow）而不是用于部署，则可以令这个参数为 False。

#### input_names, output_names

设置输入和输出张量的名称。如果不设置的话，会自动分配一些简单的名字（如数字）。 ONNX 模型的每个输入和输出张量都有一个名字。很多推理引擎在运行 ONNX 文件时，都需要以“名称-张量值”的数据对来输入数据，并根据输出张量的名称来获取输出数据。在进行跟张量有关的设置（比如添加动态维度）时，也需要知道张量的名字。 在实际的部署流水线中，我们都需要设置输入和输出张量的名称，并保证 ONNX 和推理引擎中使用同一套名称。
#### opset_version

转换时参考哪个 ONNX 算子集版本，默认为9。后文会详细介绍 PyTorch 与 ONNX 的算子对应关系。

#### dynamic_axes

指定输入输出张量的哪些维度是动态的。 为了追求效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变）。但在实际应用中，我们又希望模型的输入张量是动态的，尤其是本来就没有形状限制的全卷积模型。因此，我们需要显式地指明输入输出张量的哪几个维度的大小是可变的。
例如：0维设置为动态
```python
dynamic_axes_0 = {
    'in' : [0],
    'out' : [0]
}
```

由于 ONNX 要求每个动态维度都有一个名字，这样写的话会引出一条 UserWarning，警告我们通过列表的方式设置动态维度的话系统会自动为它们分配名字。一种显式添加动态维度名字的方法如下：
```python
dynamic_axes_0 = {
    'in' : {0: 'batch'},
    'out' : {0: 'batch'}
}
```

### 使用技巧

#### 使模型在ONNX转换时有不同的行为。
有些时候，我们希望模型在直接用 PyTorch 推理时有一套逻辑，而在导出的ONNX模型中有另一套逻辑。比如，我们可以把一些后处理的逻辑放在模型里，以简化除运行模型之外的其他代码。`torch.onnx.is_in_onnx_export()`可以实现这一任务，该函数仅在执行 `torch.onnx.export()`时为真。以下是一个例子：

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        if torch.onnx.is_in_onnx_export():
            x = torch.clip(x, 0, 1)
        return x
```

这里，我们仅在模型导出时把输出张量的数值限制在[0, 1]之间。使用 `is_in_onnx_export` 确实能让我们方便地在代码中添加和模型部署相关的逻辑。但是，这些代码对只关心模型训练的开发者和用户来说很不友好，突兀的部署逻辑会降低代码整体的可读性。同时，`is_in_onnx_export` 只能在每个需要添加部署逻辑的地方都“打补丁”，难以进行统一的管理。我们之后会介绍如何使用 MMDeploy 的重写机制来规避这些问题。

#### 利用中断张量跟踪的操作

PyTorch 转 ONNX 的跟踪导出法是不是万能的。如果我们在模型中做了一些很“出格”的操作，跟踪法会把某些取决于输入的中间结果变成常量，从而使导出的ONNX模型和原来的模型有出入。以下是一个会造成这种“跟踪中断”的例子：

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * x[0].item()
        return x, torch.Tensor([i for i in x])

model = Model()
dummy_input = torch.rand(10)
torch.onnx.export(model, dummy_input, 'a.onnx')
```

如果你尝试去导出这个模型，会得到一大堆 warning，告诉你转换出来的模型可能不正确。这也难怪，我们在这个模型里使用了`.item()`把 torch 中的张量转换成了普通的 Python 变量，还尝试遍历 torch 张量，并用一个列表新建一个 torch 张量。这些涉及张量与普通变量转换的逻辑都会导致最终的 ONNX 模型不太正确。 另一方面，我们也可以利用这个性质，在保证正确性的前提下令模型的中间结果变成常量。这个技巧常常用于模型的静态化上，即令模型中所有的张量形状都变成常量。

## PyTorch 对 ONNX 的算子支持

在确保`torch.onnx.export()`的调用方法无误后，PyTorch 转 ONNX 时最容易出现的问题就是算子不兼容了。这里我们会介绍如何判断某个 PyTorch 算子在 ONNX 中是否兼容，以助大家在碰到报错时能更好地把错误归类。而具体添加算子的方法我们会在之后的文章里介绍。 在转换普通的`torch.nn.Module`模型时，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子。在这个翻译过程中，可能会碰到以下情况：

- 该算子可以一对一地翻译成一个 ONNX 算子。
- 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。
- 该算子没有定义翻译成 ONNX 的规则，报错。

那么，该如何查看 PyTorch 算子与 ONNX 算子的对应情况呢？由于 PyTorch 算子是向 ONNX 对齐的，这里我们先看一下 ONNX 算子的定义情况，再看一下 PyTorch 定义的算子映射关系。

### ONNX 算子文档

ONNX 算子的定义情况，都可以在官方的[算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)中查看。这份文档十分重要，我们碰到任何和 ONNX 算子有关的问题都得来”请教“这份文档。
### PyTorch 对 ONNX 算子的映射
在 PyTorch 中，和 ONNX 有关的定义全部放在 [torch.onnx 目录](https://github.com/pytorch/pytorch/tree/master/torch/onnx)中，其中，`symbloic_opset{n}.py`（符号表文件）即表示 PyTorch 在支持第 n 版 ONNX 算子集时新加入的内容。我们之前讲过， bicubic 插值是在第 11 个版本开始支持的。我们以它为例来看看如何查找算子的映射情况。

# 总结
