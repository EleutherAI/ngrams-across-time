from dataclasses import dataclass

from tqdm import tqdm
from datasets import ClassLabel, DatasetDict, Features, Image, load_dataset # Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from concept_erasure import QuadraticEraser
from concept_erasure.utils import assert_type
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2.functional as TF
from argparse import ArgumentParser

from ngrams_across_time.utils.utils import set_seeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mean_std(X):
    channels_mean = torch.mean(X, dim=[0,2,3])
    channels_squared_sum = torch.mean(X**2, dim=[0,2,3])

    std = (channels_squared_sum - channels_mean ** 2) ** 0.5
    
    return channels_mean, std


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size * 4) # input_size * 8 # 4x?
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size * 4, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


class HfDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'pixel_values': self.data[idx],
            'label': self.labels[idx]
        }   


def run_dataset(dataset_str: str, seed: int, edited: bool):
    images = None
    labels = None
    
    if edited:
        def load_edited_dataset(filename):
            loaded_data = torch.load(filename)
            return HfDataset(loaded_data['data'], loaded_data['labels'])
        train_dataset = load_edited_dataset('edited_train_dataset.pth')
        test_dataset = load_edited_dataset('edited_test_dataset.pth')
        
        images = train_dataset.data.to(device)
        labels = train_dataset.labels.to(device)
        
        test_images = test_dataset.data.to(device)
        test_labels = test_dataset.labels.to(device)
    else:
        path, _, name = dataset_str.partition(":")
        ds: DatasetDict = load_dataset(path, name or None) # type: ignore
        train_dataset = ds['train'].with_format("torch")
        test_dataset = ds['test'].with_format("torch")

        images =  assert_type(Tensor, train_dataset['img']).div(255).to(device)
        labels = assert_type(Tensor, train_dataset['label']).to(device)

        test_images = assert_type(Tensor, test_dataset['img']).div(255).to(device)
        test_labels = assert_type(Tensor, test_dataset['label']).to(device)

        # normalize images to mean and std of 0.5
        mean, std = get_mean_std(images)
        images = TF.normalize(images, mean.tolist(), std.tolist())
        mean, std = get_mean_std(test_images)
        test_images = TF.normalize(test_images, mean.tolist(), std.tolist())


    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 20_000
    learning_rate = 1e-3

    set_seeds(seed)

    # Train the model
    input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels
    num_classes = 10
    model = MLP(input_size, num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs)):
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            with torch.no_grad():
                outputs = model(test_images)
                _, predicted = torch.max(outputs.data, 1)
                total = test_labels.size(0)
                correct = (predicted == test_labels).sum().item()

            print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
            print(f'Loss of the model on the 10000 test images: {loss_fn(outputs, test_labels).item()}')

    # Save the model
    torch.save({
        'edited': edited,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }, f'mlp_cifar10_{"edited" if edited else "not_edited"}.pth')


# Build second order concept edited dataset
def build_dataset(dataset_str: str):
    @dataclass
    class ErasedDataset(Dataset):
        eraser: QuadraticEraser
        X: Tensor
        Y: Tensor

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            x, y = self.X[idx], int(self.Y[idx])
            x = self.eraser.optimal_transport(y, x.unsqueeze(0)).squeeze(0)
            
            return {
                "pixel_values": x,
                "label": torch.tensor(y),
            }

        def __len__(self) -> int:
            return len(self.Y)

    def save_edited_dataset(dataset, filename, batch_size=256):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_data = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc=f"Processing {filename}"):
            all_data.append(batch['pixel_values'])
            all_labels.append(batch['label'])
        
        edited_data = torch.cat(all_data)
        edited_labels = torch.cat(all_labels)
        
        torch.save({
            'data': edited_data,
            'labels': edited_labels
        }, filename)
        
        print(f"Saved edited dataset to {filename}")

    # Allow specifying load_dataset("svhn", "cropped_digits") as "svhn:cropped_digits"
    # We don't use the slash because it's a valid character in a dataset name
    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["train"][0][img_col]
    c, (h, w) = len(example.mode), example.size
    print(f"Image size: {h} x {w}")

    train = ds["train"].with_format("torch")
    X_train: Tensor = assert_type(Tensor, train[img_col]).div(255)
    Y_train = assert_type(Tensor, train[label_col])

    test = ds["test"].with_format("torch")
    X_test: Tensor = assert_type(Tensor, test[img_col]).div(255)
    Y_test = assert_type(Tensor, test[label_col])

    eraser = QuadraticEraser.fit(X_train.flatten(1, 3), Y_train)
    edited_ds_train = ErasedDataset(eraser, X_train, Y_train)
    edited_ds_test = ErasedDataset(eraser, X_test, Y_test)

    save_edited_dataset(edited_ds_train, 'edited_train_dataset.pth')
    save_edited_dataset(edited_ds_test, 'edited_test_dataset.pth')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--edited", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # build_dataset(args.dataset)
    run_dataset(args.dataset, args.seed, args.edited)