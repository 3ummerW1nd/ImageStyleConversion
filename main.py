import torch.optim as optim
from models import *
import torchvision.models as models
import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 512
style_img_name = input("请输入风格图片的文件名：")
content_img_name = input("请输入内容图片的文件名：")
print("请稍候")
style_img = read_image('style_imgs/' + style_img_name, target_width=width).to(device)
content_img = read_image('content_imgs/' + content_img_name, target_width=width).to(device)
plt.figure(figsize=(12, 6))
vgg19 = models.vgg19(pretrained=True)
print(vgg19)
vgg19 = VGG(vgg19.features[:30]).to(device).eval()

style_features = vgg19(style_img)
content_features = vgg19(content_img)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


style_grams = [gram_matrix(x) for x in style_features]

input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])
style_weight = 1e6
content_weight = 1

run = [0]
while run[0] <= 300:
    def f():
        optimizer.zero_grad()
        features = vgg19(input_img)
        content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
        style_loss = 0
        grams = [gram_matrix(x) for x in features]
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight
        loss = style_loss + content_loss
        run[0] += 1
        loss.backward()
        return loss
    optimizer.step(f)
plt.figure(figsize=(18, 6))
imsave(input_img, style_img_name, content_img_name)
print("已完成")