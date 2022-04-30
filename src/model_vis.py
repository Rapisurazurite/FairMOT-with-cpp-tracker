from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from trains.mot import MotTrainer

from torchviz import make_dot
from models.networks.pose_dla_dcn import dla34


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)

    """
        vis the dataset
        Key of dataset:
            input: [3, 608, 1088]
            hm: [1, 152, 272] -> gauss heatmap
            reg_mask: [500] -> ignore
            ind: [500] -> index of object in the image pixel 
            wh: [500, 4]  -> (l, t, r, b)
            reg: [500, 2] -> offset of the center of object
            ids: [500] -> label of object
            bbox: [500, 4] -> x1, y1, x2, y2
    """
    if False:
        hm = dataset[0]['hm']
        import matplotlib.pyplot as plt
        import sys
        plt.imshow(hm[0, :, :])
        plt.title('heatmap')
        plt.show()

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    for k, v in opt.__dict__.items():
        print(k, ':', v)
    # print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    print('Starting training...')
    trainer = MotTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    fake_data = torch.randn(1, 3, 608, 1088)
    fake_data = fake_data.cuda()

    dla_34 = dla34(pretrained=True).cuda()
    print(dla_34)
    out = dla_34(fake_data)
    out_tuple = tuple(out)
    g = make_dot(out_tuple, params=dict(dla_34.named_parameters()), show_attrs=True)
    g.render("dla34.pdf", view=False)


    # g = make_dot(out_tuple, params=dict(model.named_parameters()), show_attrs=True)


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
