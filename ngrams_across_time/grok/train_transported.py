from dataclasses import dataclass

from tqdm import tqdm
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from concept_erasure import QuadraticEraser
from concept_erasure.utils import assert_type
from torch.utils.data import DataLoader, Dataset

from ngrams_across_time.utils.utils import set_seeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]

@dataclass
class ErasedDataset(Dataset):
    eraser: QuadraticEraser
    X: Tensor
    Y: Tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], int(self.Y[idx])
        x_erased = self.eraser.optimal_transport(y, x.unsqueeze(0)).squeeze(0)
        x = x_erased.reshape(x.shape)
        
        return {
            "pixel_values": x,
            "label": torch.tensor(y),
        }

    def __len__(self) -> int:
        return len(self.Y)



class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size * 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size * 8, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out

class EditedDataset(Dataset):
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

def run_dataset(dataset_str: str, seed: int):
    def load_edited_dataset(filename):
        loaded_data = torch.load(filename)
        return EditedDataset(loaded_data['data'], loaded_data['labels'])

    edited_ds_train = load_edited_dataset('edited_train_dataset.pth')
    edited_ds_test = load_edited_dataset('edited_test_dataset.pth')

    num_epochs = 20_000
    learning_rate = 1e-3

    set_seeds(seed)

    # Allow specifying load_dataset("svhn", "cropped_digits") as "svhn:cropped_digits"
    # We don't use the slash because it's a valid character in a dataset name
    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    # CIFAR-10 dataset
    # train_dataset = torchvision.datasets.CIFAR10(root='.', train=True, download=True) # transform=transform
    # test_dataset = torchvision.datasets.CIFAR10(root='.', train=False, download=True) # transform=transform
    # train_loader = DataLoader(train_dataset, batch_size=len(edited_ds_train) // 4, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=len(edited_ds_test) // 4, shuffle=False)

    train_loader = DataLoader(edited_ds_train, batch_size=len(edited_ds_train), shuffle=True)
    test_loader = DataLoader(edited_ds_test, batch_size=len(edited_ds_test), shuffle=False)

    # Train the model
    input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels
    num_classes = 10
    model = MLP(input_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    images = edited_ds_train.data.to(device)
    labels = edited_ds_train.labels.to(device)

    for epoch in tqdm(range(num_epochs)):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            # model.eval()
            with torch.no_grad():
                test_images = edited_ds_test.data.to(device)
                test_labels = edited_ds_test.labels.to(device)
                outputs = model(test_images)
                _, predicted = torch.max(outputs.data, 1)
                total = test_labels.size(0)
                correct = (predicted == test_labels).sum().item()

            print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

    # Save the model
    torch.save(model.state_dict(), 'mlp_cifar10.pth')


# ConceptEditedDataset produces data on the fly so we loop over Concept Edited Dataset for training and testing and save with torch.save
# Save the edited datasets
def build_dataset(dataset_str: str):
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

    save_edited_dataset(edited_ds_train, 'edited_train_dataset.pth')
    save_edited_dataset(edited_ds_test, 'edited_test_dataset.pth')

if __name__ == "__main__":
    # build_dataset("cifar10")
    run_dataset("cifar10", 1)