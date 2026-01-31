模型部署时一般会碰到以下几类困难：

- 模型的动态化。出于性能的考虑，各推理框架都默认模型的输入形状、输出形状、结构是静态的。而为了让模型的泛用性更强，部署时需要在尽可能不影响原有逻辑的前提下，让模型的输入输出或是结构动态化。
- 新算子的实现。深度学习技术日新月异，提出新算子的速度往往快于 ONNX 维护者支持的速度。为了部署最新的模型，部署工程师往往需要自己在 ONNX 和推理引擎中支持新算子。
- 中间表示与推理引擎的兼容问题。由于各推理引擎的实现不同，对 ONNX 难以形成统一的支持。为了确保模型在不同的推理引擎中有同样的运行效果，部署工程师往往得为某个推理引擎定制模型代码，这为模型部署引入了许多工作量。
# 模型的动态化
1. 动态参数 -> 推理输入，且用torch.tensor包装

# 新算子实现
```python
class NewInterpolate(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, scales):
        return g.op("Resize",
                    input,
                    g.op("Constant",
                         value_t=torch.tensor([], dtype=torch.float32)),
                    scales,
                    coordinate_transformation_mode_s="pytorch_half_pixel",
                    cubic_coeff_a_f=-0.75,
                    mode_s='cubic',
                    nearest_mode_s="floor")

    @staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(input,
                           scale_factor=scales,
                           mode='bicubic',
                           align_corners=False)
```
我们希望新的插值算子有两个输入，一个是被用于操作的图像，一个是图像的放缩比例。前面讲到，为了对接 ONNX 中 Resize 算子的 scales 参数，这个放缩比例是一个 [1, 1, x, x] 的张量，其中 x 为放大倍数。在之前放大3倍的模型中，这个参数被固定成了[1, 1, 3, 3]。因此，在插值算子中，我们希望模型的第二个输入是一个 [1, 1, w, h] 的张量，其中 w 和 h 分别是图片宽和高的放大倍数。

搞清楚了插值算子的输入，再看一看算子的具体实现。算子的推理行为由算子的 forward 方法决定。该方法的第一个参数必须为 ctx，后面的参数为算子的自定义输入，我们设置两个输入，分别为被操作的图像和放缩比例。为保证推理正确，需要把 [1, 1, w, h] 格式的输入对接到原来的 interpolate 函数上。我们的做法是截取输入张量的后两个元素，把这两个元素以 list 的格式传入 interpolate 的 scale_factor 参数。

接下来，我们要决定新算子映射到 ONNX 算子的方法。映射到 ONNX 的方法由一个算子的 symbolic 方法决定。symbolic 方法第一个参数必须是g，之后的参数是算子的自定义输入，和 forward 函数一样。ONNX 算子的具体定义由 g.op 实现。g.op 的每个参数都可以映射到 ONNX 中的算子属性：
![[Pasted image 20240111002600.png]]
```python
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model, (x, factor),
                      "srcnn3.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
```
# 总结
- 模型部署中常见的几类困难有：模型的动态化；新算子的实现；框架间的兼容。
- PyTorch 转 ONNX，实际上就是把每一个操作转化成 ONNX 定义的某一个算子。比如对于 PyTorch 中的 Upsample 和 interpolate，在转 ONNX 后最终都会成为 ONNX 的 Resize 算子。
- 通过修改继承自 torch.autograd.Function 的算子的 symbolic 方法，可以改变该算子映射到 ONNX 算子的行为。
