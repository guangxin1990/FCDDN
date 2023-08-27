import os
import time
import datetime
import torch
import transforms as T

from FCDDN import FCDDN
from train_utils import evaluate
from my_dataset import BreastDataset

class SegmentationPresetEval:
    def __init__(self, mean=(0.261, 0.261, 0.261), std=(0.134, 0.134, 0.134)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(mean=(0.261, 0.261, 0.261), std=(0.134, 0.134, 0.134)):
    return SegmentationPresetEval(mean=mean, std=std)

def create_model(num_classes):
    model = FCDDN(n_classes=num_classes)
    return model

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    num_classes = args.num_classes

    mean = (0.261, 0.261, 0.261)
    std = (0.134, 0.134, 0.134)

    results_file = "./result/results_test{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    val_dataset = BreastDataset(args.data_path,
                               transforms=get_transform(mean=mean, std=std), txt_name="val.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)

    model.to(device)

    model.load_state_dict(torch.load('./save_weights/best_model.pth')['model'])

    start_time = time.time()

    confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
    val_info = str(confmat)
    print(val_info)

    with open(results_file, "a") as f:
        f.write(val_info + "\n\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("testing time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcddn test")
    parser.add_argument("--data-path", default="./", help="BreastData root")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="testing device")
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
