import os
import time
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from FCDDN import FCDDN


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "./save_weights/best_model.pth"
    img_path = "./data/images/benign (1).png"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    mean = (0.261, 0.261, 0.261)
    std = (0.134, 0.134, 0.134)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = FCDDN(n_classes=2)

    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    original_img = Image.open(img_path).convert('RGB')

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():

        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        prediction[prediction == 1] = 255
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
