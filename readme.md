# VAE-PyTorch

这是一个使用PyTorch实现的变分自编码器(VAE)项目。

## 项目描述

本项目实现了一个基本的变分自编码器(VAE)，用于处理MNIST手写数字数据集。VAE是一种生成模型，能够学习数据的压缩表示并生成新的样本。

## 功能特点

- 使用PyTorch框架实现VAE模型
- 包含编码器、解码器和重参数化技巧
- 使用MNIST数据集进行训练
- 结合重构误差(BCE)和KL散度的损失函数
- 支持GPU加速（如果可用）

## 环境要求

- Python 3.6+
- PyTorch
- torchvision

## 安装依赖

```bash
pip install torch torchvision
```

## 使用方法

1. 克隆仓库：

```bash
git clone https://github.com/您的用户名/vae-pytorch.git
cd vae-pytorch
```

2. 运行主程序：

```bash
python main.py
```

## 代码结构

- `main.py`: 包含VAE模型定义、训练循环和主函数

## 模型架构

- 输入维度：784 (28x28 MNIST图像)
- 隐藏层维度：400
- 潜在空间维度：20

## 训练细节

- 优化器：Adam
- 学习率：1e-3
- 批次大小：128
- 训练轮数：10 epochs

## 注意事项

- 如果有CUDA设备可用，模型会自动使用GPU进行训练
- 可以通过调整`main.py`中的参数来修改模型架构和训练设置

## 未来改进

- 添加模型评估和图像生成功能
- 实现更复杂的VAE变体
- 添加可视化工具以展示训练过程和生成结果

## 贡献

欢迎提出问题、建议或直接贡献代码。请随时创建issue或提交pull request。

## 许可证

本项目采用MIT许可证。详情请见[LICENSE](LICENSE)文件。
