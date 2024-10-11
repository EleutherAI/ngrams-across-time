from dataclasses import dataclass
from pathlib import Path
import pickle
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
import torchvision.transforms.v2.functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
# import torchvision.transforms as transforms
from concept_erasure import QuadraticEditor, QuadraticFitter
from concept_erasure.utils import assert_type
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from ngrams_across_time.utils.utils import set_seeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ConceptEditedDataset(Dataset):
    class_probs: Tensor
    editor: QuadraticEditor
    X: Tensor
    Y: Tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], int(self.Y[idx])

        # Make sure we don't sample the correct class
        loo_probs = self.class_probs.clone()
        loo_probs[y] = 0
        target_y = torch.multinomial(loo_probs, 1).squeeze()

        x = self.editor.transport(x[None], y, int(target_y)).squeeze(0)
        return {
            "pixel_values": x,
            "label": target_y,
        }

    def __len__(self) -> int:
        return len(self.Y)


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

    num_epochs = 50
    batch_size = 256
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
    # train_dataset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform)
    # test_dataset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(edited_ds_train, batch_size=len(edited_ds_train) // 4, shuffle=True)
    test_loader = DataLoader(edited_ds_test, batch_size=len(edited_ds_test) // 4, shuffle=False)

    # Train the model
    input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels
    num_classes = 10
    model = MLP(input_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in tqdm(range(num_epochs)):
        for i, batch in enumerate(train_loader):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    X_train = rearrange(X_train, "n h w c -> n c h w")
    Y_train = assert_type(Tensor, train[label_col])

    test = ds["test"].with_format("torch")
    X_test: Tensor = assert_type(Tensor, test[img_col]).div(255)
    X_test = rearrange(X_test, "n h w c -> n c h w")
    Y_test = assert_type(Tensor, test[label_col])

    print("Computing statistics...")
    fitter_path = Path('fitter')
    fitter_path.mkdir(exist_ok=True)
    if (fitter_path / f"{dataset_str}_train.pth").exists():
        fitter_train = torch.load(fitter_path / f"{dataset_str}_train.pth")
        fitter_test = torch.load(fitter_path / f"{dataset_str}_test.pth")
    else:
        fitter_train = QuadraticFitter.fit(X_train.flatten(1).cuda(), Y_train.cuda())
        torch.save(fitter_train, fitter_path / f"{dataset_str}_train.pth")
        fitter_test = QuadraticFitter.fit(X_test.flatten(1).cuda(), Y_test.cuda())
        torch.save(fitter_test, fitter_path / f"{dataset_str}_test.pth")
    

    class_probs = torch.bincount(Y_train).float()
        
    cache = Path('/mnt/ssd-1/lucia/features-across-time') / "editor-cache" / f"{dataset_str}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            editor = pickle.load(f)
    else:
        print("Computing optimal transport maps...")

        editor = fitter_train.editor("cpu")
        cache.parent.mkdir(exist_ok=True)

        with open(cache, "wb") as f:
            pickle.dump(editor, f)

    edited_ds_train = ConceptEditedDataset(class_probs, editor, X_train, Y_train)
    edited_ds_test = ConceptEditedDataset(class_probs, editor, X_test, Y_test)

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
    build_dataset("cifar10")
    run_dataset("cifar10", 1)



# def preprocess(batch):
#     return {
#         "pixel_values": [TF.to_tensor(x) for x in batch[img_col]],
#         "label": torch.tensor(batch[label_col]),
#     }

# if val := ds.get("validation"):
#     test = ds["test"].with_transform(preprocess) if "test" in ds else None
#     val = val.with_transform(preprocess)
# else:
#     nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
#     val = nontrain["train"].with_transform(preprocess)
#     test = nontrain["test"].with_transform(preprocess)
