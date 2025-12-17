import torch
print(torch.__version__)          # 查看PyTorch版本
print(torch.cuda.is_available())  # 应输出True
print(torch.version.cuda)