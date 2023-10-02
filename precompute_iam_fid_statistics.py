import argparse
import os
from pathlib import Path
import pickle
import PIL

import numpy as np
import torch
from datadings.reader import MsgpackReader
from torch.utils.data import Dataset
import io
from PIL import Image
from dnnlib.util import open_url
from utils.util import get_activations, set_seeds
import numpy

x = numpy.random.rand(100, 5)
numpy.random.shuffle(x)
training, test = x[:80, :], x[80:, :]


class BaseIAMDataset(Dataset):
    def __init__(self, data_path, split="train", seed=42):
        self.dataset = MsgpackReader(data_path)
        base_path = Path("dataset_splits/iam_dataset/")
        if not (base_path / f"{split}.npy").exists():
            base_path.mkdir(parents=True, exist_ok=True)

            indices = np.arange(len(self.dataset))
            np.random.seed(seed)
            np.random.shuffle(indices)
            train, test = (
                indices[: int(0.9 * (len(self.dataset)))],
                indices[int(0.9 * (len(self.dataset))) :],
            )
            np.save(base_path / f"train.npy", train)
            np.save(base_path / f"test.npy", test)
        self.split_indices = np.load(base_path / f"{split}.npy")

    def __getitem__(self, index):
        sample = self.dataset[self.split_indices[index]]
        image = Image.open(io.BytesIO(sample["image"]["bytes"]))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return image, 0

    def __len__(self):
        return len(self.split_indices)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_dir = Path(args.fid_dir)
    fid_dir.mkdir(parents=True, exist_ok=True)
    file_path = fid_dir / f"{args.split}.npz"
    dataset = BaseIAMDataset(args.data_path, split=args.split)
    queue = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)

    with open_url(
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    ) as f:
        model = pickle.load(f).to(device)

    act = get_activations(queue, model, device=device, max_samples=len(queue.dataset))
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    np.savez(file_path, mu=mu, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size per GPU"
    )
    parser.add_argument(
        "--fid_dir",
        type=str,
        default="./fid_stats/iam_dataset",
        help="A dir to store fid related files",
    )
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    set_seeds(0, 0)

    main(args)
