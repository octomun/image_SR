import argparse
import os
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import RCAN
from inference import inference
from dataset import test_Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2
import zipfile

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--test_images_dir', type=str, default='../data/test/lr')
    parser.add_argument('--output_size', type=int, default=2048)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    model = RCAN(opt)

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    test_dataset = test_Dataset(opt.test_images_dir, opt.patch_size, opt.scale, opt.use_fast_loader)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.threads)

    pred_img_list, pred_name_list = inference(model, test_dataloader, device)

    os.makedirs('../submission', exist_ok=True)
    sub_imgs = []
    for path, pred_img in tqdm(zip(pred_name_list, pred_img_list)):
        cv2.imwrite('../submission'+path, pred_img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile("../submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()
    print('Done.')