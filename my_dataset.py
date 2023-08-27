import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    def __init__(self, root: str, transforms=None, txt_name: str = "train.txt"):
        super(BreastDataset, self).__init__()
        data_root = os.path.join(root, "data")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."

        txt_path = os.path.join(data_root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.img_list = [os.path.join(data_root, "images", x) for x in file_names]
        self.mask = [os.path.join(data_root, "mask", x.split(".")[0] + "_mask.png") for x in file_names]

        for i in self.img_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        mask = 255 - np.array(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] > 128:
                    mask[i, j] = 255
                else:
                    mask[i, j] = 0
        mask = mask / 255
        roi_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        roi_mask[120:450, 200:600] = 255
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(mask + roi_mask, a_min=0, a_max=255)

        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



