import yaml
from pathlib import Path
from torchvision import datasets

# Load paths
with open("./configs/paths.yaml", "r") as f :
    paths = yaml.safe_load(f)

DATA_DIR = Path(paths["data_dir"])

# Download CIFAR-10 dataset
def download_cifar10(data_dir = DATA_DIR) :
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)

if __name__ == "__main__" :
    download_cifar10()