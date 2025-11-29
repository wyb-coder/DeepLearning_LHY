import torchvision.models as models

model = models.resnet18(weights=None)  # 不加载预训练权重
print(model)
