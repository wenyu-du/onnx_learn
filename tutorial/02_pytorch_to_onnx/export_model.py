import torch
import torch.nn as nn
import torch.onnx

# 1. 定义一个简单的模型
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

def export():
    # 2. 实例化模型并加载权重（这里使用随机权重）
    model = SuperResolutionNet(upscale_factor=3)
    model.eval()

    # 3. 准备示例输入
    x = torch.randn(1, 1, 224, 224, requires_grad=True)

    # 4. 导出模型
    torch.onnx.export(model,               # 模型实例
                      x,                   # 模型输入
                      "super_resolution.onnx", # 导出文件名
                      export_params=True,  # 存储训练好的权重参数
                      opset_version=11,    # ONNX 算子集版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names = ['input'],   # 输入节点名称
                      output_names = ['output'], # 输出节点名称
                      dynamic_axes={'input' : {0 : 'batch_size'},    # 动态轴（可选）
                                    'output' : {0 : 'batch_size'}})

    print("Model has been converted to ONNX")

if __name__ == "__main__":
    export()
