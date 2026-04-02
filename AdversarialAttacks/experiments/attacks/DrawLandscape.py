from data import get_NIPS17_loader
from attacks import BIM, MI_CommonWeakness, MI_FGSM
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import MultipleLocator

y_major_locator = MultipleLocator(1.0)
y_formatter = FormatStrFormatter("%1.1f")
ax = plt.gca()
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_major_locator(y_major_locator)

font1 = {
    "family": "Times New Roman",
    "weight": "bold",
    "style": "normal",
    "size": 15,
}
font2 = {
    "family": "Times New Roman",
    "weight": "bold",
    "style": "normal",
}

loader = get_NIPS17_loader(batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# origin_train_models = [resnet152]
# origin_test_models = [resnet18]
train_models = [
    BaseNormModel(resnet18(pretrained=True)).eval(),
    BaseNormModel(resnet34(pretrained=True)).eval(),
    BaseNormModel(resnet50(pretrained=True)).eval(),
    BaseNormModel(resnet101(pretrained=True)).eval(),
    Salman2020Do_R50().eval(),
    Debenedetti2022Light_XCiT_S12().eval(),
]
for m in train_models:
    m.eval()
    m.requires_grad_(False)
origin_test_models = [
    Engstrom2019Robustness,
    Salman2020Do_50_2,
    Debenedetti2022Light_XCiT_M12,
    Debenedetti2022Light_XCiT_L12,
]

test_models = []

for model in origin_test_models:
    model = Identity(model(pretrained=True)).to(device)
    model.eval()
    model.requires_grad_(False)
    test_models.append(model)
# test_models = train_models

# test_models += train_models

attacker = MI_FGSM(train_models, targeted_attack=True)
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
original_x = x.clone()
x = attacker(x, y)
test_models = [m.to(torch.device("cuda")) for m in test_models]
for i in range(len(test_models)):
    finetuner = BIM([test_models[i]], step_size=1e-7, targeted_attack=True, total_step=100)
    now_x = finetuner(x.clone(), y)
    drawer = Landscape4Input(lambda x: F.cross_entropy(test_models[i](x), y.cuda()).item(), input=x.cuda(), mode="2D")
    drawer.synthesize_coordinates(x_min=-15 / 255, x_max=15 / 255, x_interval=1 / 255)
    direction = (now_x - x) * (-1) ** i
    direction /= torch.max(torch.abs(direction))  # normalize by ell_inf norm
    drawer.assign_unit_vector(direction)
    drawer.draw()
legends = [
    "ResNet-50 (Engstrom et al., 2019)",
    "ResNet-50 (Salman et al., 2020)",
    "XCiT-M12 (Debenedetti et al., 2022)",
    "XCiT-L12 (Debenedetti et al., 2022)",
]
plt.legend(legends, prop=font2, loc=1)
plt.xlabel("Distance under Infinity Norm", fontdict=font1)
plt.ylabel("Loss", fontdict=font1)
plt.yticks(fontproperties="Times New Roman", size=15)
plt.xticks(fontproperties="Times New Roman", size=15)
plt.ylim(0, 5)
plt.savefig("./MI-FGSM.pdf", bbox_inches="tight")
