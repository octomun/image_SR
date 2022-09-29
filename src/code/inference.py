from torchvision import transforms
import torch
from tqdm import tqdm
import PIL.Image as pil_image

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    pred_img_list = []
    name_list = []
    with torch.no_grad():
        for lr_img, file_name in tqdm(iter(test_loader)):

            pred = model(lr_img)
            output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            output = pil_image.fromarray(output, mode='RGB')

            pred_img_list.append(output)
            name_list.append(file_name)
    return pred_img_list, name_list