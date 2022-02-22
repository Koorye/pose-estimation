import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from visdom import Visdom

from models.hourglass import hg_stack2
from models.pose_res_net import PoseResNet
from models.hr_net import hr_w32
from joints_mse_loss import JointsMSELoss
from mpii_dataset import MPIIDataset

from utils import heatmaps2rgb

seed = 999
use_model = 'Hourglass_Stack2' # 可选：Hourglass_Stack2, ResNet, HRNet
lr = 1e-3
bs = 12
n_epoches = 20
# ckpt = 'weights/Hourglass_Stack2_epoch1.pth' # 历史模型文件
ckpt = None

print(f'Use Model: {use_model}')
if ckpt:
    print(f'Load ckpt {ckpt}')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0')

dataset = MPIIDataset(use_scale=True, use_flip=True, use_rand_color=True)
data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

if use_model == 'Hourglass_Stack2':
    model = hg_stack2().to(device)
elif use_model == 'ResNet':
    model = PoseResNet().to(device)
elif use_model == 'HRNet':
    model = hr_w32(32).to(device)
else:
    raise NotImplementedError

optimizer = Adam(model.parameters(), lr=lr)
lr_scheduler = MultiStepLR(optimizer, [10,15], .1)
criteon = JointsMSELoss().to(device)

ep_start = 1
if ckpt:
    weight_dict = torch.load(ckpt)
    model.load_state_dict(weight_dict['model'])
    optimizer.load_state_dict(weight_dict['optim'])
    lr_scheduler.load_state_dict(weight_dict['lr_scheduler'])
    ep_start = weight_dict['epoch'] + 1

target_weight = np.array([[1.2, 1.1, 1., 1., 1.1, 1.2, 1., 1.,
                           1., 1., 1.2, 1.1, 1., 1., 1.1, 1.2]])
target_weight = torch.from_numpy(target_weight).to(device).float()

viz = Visdom()
viz.line([0], [0], win='Train Loss', opts=dict(title='Train Loss'))

for ep in range(ep_start, n_epoches+1):

    total_loss, count = 0., 0

    for index, (img, heatmaps, pts) in enumerate(tqdm(data_loader, desc=f'Epoch{ep}')):
        img, heatmaps = img.to(device).float(), heatmaps.to(device).float()

        if use_model in ['ResNet', 'HRNet']:
            heatmaps_pred = model(img)
            loss = criteon(heatmaps_pred, heatmaps, target_weight)

        elif use_model in ['Hourglass_Stack2']:
            heatmaps_preds = model(img)
            heatmaps_pred = heatmaps_preds[-1]
            # 中继监督
            loss1 = criteon(heatmaps_preds[0], heatmaps, target_weight)
            loss2 = criteon(heatmaps_preds[1], heatmaps, target_weight)
            loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_step = (ep-1) * len(data_loader) + index
        total_loss += loss.item()
        count += 1
        if count == 10 or index == len(data_loader) - 1:
            viz.line([total_loss / count], [cur_step], win='Train Loss', update='append')
            viz.image(img[0], win='Image', opts=dict(title='Image'))
            viz.images(heatmaps2rgb(heatmaps[0]), nrow=4,
                       win=f'GT Heatmaps', opts=dict(title=f'GT Heatmaps'))
            viz.images(heatmaps2rgb(heatmaps_pred[0]), nrow=4,
                       win=f'Pred Heatmaps', opts=dict(title=f'Pred Heatmaps'))
            
            final_loss = total_loss / count
            total_loss, count = 0., 0

    lr_scheduler.step()

    torch.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, f'weights/{use_model}_epoch{ep}_loss{final_loss}.pth')

    torch.cuda.empty_cache()
