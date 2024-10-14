from argparse import ArgumentParser
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
from datasets import Dataset as HfDataset, load_dataset
from tqdm import tqdm
import lovely_tensors as lt
from sae import SaeConfig, SaeTrainer, TrainConfig

from ngrams_across_time.utils.utils import set_seeds, assert_type


def create_balanced_sample(dataset: HfDataset, n):
    dataset.set_format(type=None, columns=['pixel_values', 'label'])
    num_classes = len(set(dataset['label']))
    samples_per_class = n // num_classes

    class_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(dataset['label']):
        class_indices[label].append(idx)
    
    # Randomly select samples_per_class indices from each class
    balanced_indices = []
    for class_idx in class_indices:
        if len(class_indices[class_idx]) >= samples_per_class:
            balanced_indices.extend(random.sample(class_indices[class_idx], samples_per_class))
        else:
            # If a class has fewer samples than required, take all available and randomly oversample
            balanced_indices.extend(class_indices[class_idx])
            balanced_indices.extend(random.choices(class_indices[class_idx], 
                                                   k=samples_per_class - len(class_indices[class_idx])))


    balanced_dataset = HfDataset.from_dict({
        'pixel_values': [dataset['pixel_values'][i] for i in balanced_indices],
        'label': [dataset['label'][i] for i in balanced_indices]
    })
    balanced_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
    return balanced_dataset
    

def mean_std(dataset: HfDataset, batch_size=1000, num_workers=4, num_channels=1):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers) # type: ignore

    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    total_samples = 0

    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std


def parse_args():
    parser = ArgumentParser(description="Train Vision Transformer on MNIST")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_name", type=str, default='')
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lt.monkey_patch()
args = parse_args()
set_seeds(args.seed)

# Define hyperparameters
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
checkpoint_interval = 5

from datasets import load_dataset
from torchvision import transforms
import torch

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    return transform(image)

dataset = load_dataset("mnist", split='train')
dataset = dataset.map(
    lambda example: {"pixel_values": transform_image(example['image'])},
    new_fingerprint='transformed_mnist'
)
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

pixel_values = assert_type(torch.Tensor, dataset['pixel_values'])
mean = pixel_values.mean().item()
std = pixel_values.std().item()
def full_transform(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    return transform(image)

dataset = dataset.map(
    lambda example: {"pixel_values": full_transform(example['image'])},
    new_fingerprint='transformed_mnist'
)

# Induce overfitting by training on a small subset of the train set
train_dataset = create_balanced_sample(dataset, 40)

test_dataset = dataset = load_dataset("mnist", split='test')
test_dataset = test_dataset.map(
    lambda example: {"pixel_values": full_transform(example['image'])},
    remove_columns=['image'],
    new_fingerprint='transformed_mnist'
)
test_dataset.set_format(type='torch', columns=['pixel_values', 'label'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # type: ignore
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # type: ignore

config = ViTConfig(
    image_size=32,
    patch_size=4,
    num_channels=1,
    num_labels=10,
    num_hidden_layers=4,
    hidden_size=256,
    num_attention_heads=4,
    intermediate_size=4096,
)
model = ViTForImageClassification(config).to(device)

criterion = nn.CrossEntropyLoss()
# Induce overfitting by zeroing weight decay
wd = 0.
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)

run_identifier = f"mnist_seed_{args.seed}{'_' + args.run_name if args.run_name else ''}"
checkpoints_path = Path("vit_ckpts") / run_identifier
checkpoints_path.mkdir(exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data).logits
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

    train_accuracy = 100.0 * train_correct / train_total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = checkpoints_path / f"mnist_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Train an SAE
        run_name = Path(f"sae/{run_identifier}/{epoch}")
        if run_name.exists():
            continue
            
        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=16,
            run_name=str(run_name),
            log_to_wandb=True,
            layers=[0, 1, 2, 3],
            dummy_inputs={'pixel_values': torch.randn(3, 1, 32, 32)},
        )

        trainer = SaeTrainer(cfg, train_dataset, model)
        trainer.fit()

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data).logits
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

print("Training completed!")