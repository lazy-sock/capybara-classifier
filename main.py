import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

# Data
class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform

        # Walk through all subdirectories and collect image paths
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only images
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = img_path.split('.')[0].split("/")[-1]
        label = int(label) - 1  # Adjust label to start from 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    

# Transformations
normalize = transforms.Normalize(mean=[0.47667825,0.46205,0.38689211], std=[0.20853477,0.20311771,0.20317237])
transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    # transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder('bird_dataset_v3', transform=transform)

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 11)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 64 * 8 * 8)
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(epoch, train_loader, model=model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

from sklearn.metrics import classification_report

def test(test_loader, model=model):
    model.eval()
    correct_per_class = torch.zeros(11, dtype=torch.int32).to(device)
    total_per_class = torch.zeros(11, dtype=torch.int32).to(device)
    class_names = [f"Class {i+1}" for i in range(11)]  # Replace with actual class names if available

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)

            for i in range(len(target)):
                label = target[i].item()
                total_per_class[label] += 1
                if preds[i] == label:
                    correct_per_class[label] += 1

    accuracies = correct_per_class.float() / total_per_class.float()
    accuracy_list = [(class_names[i], accuracies[i].item() * 100 if total_per_class[i] > 0 else 0) for i in range(10)]

    # Sort by accuracy in descending order
    accuracy_list.sort(key=lambda x: x[1], reverse=True)

    overall_accuracy = correct_per_class.sum().item() / total_per_class.sum().item() * 100
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")

    # Write sorted accuracy results to a file
    with open("accuracy_results.txt", "w") as f:
        f.write(f"Overall Test Accuracy: {overall_accuracy:.2f}%\n\n")
        for class_name, acc in accuracy_list:
            f.write(f"{class_name}: {acc:.2f}% accuracy\n")

    print("Class-wise accuracy results saved to 'accuracy_results.txt'")

def first_training(epochs):
    current_epoch = 0

    # Splitting the dataset
    train_size = int(0.7 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if __name__ == "__main__":
        for epoch in range(1, epochs + 1):
            current_epoch += 1
            train(current_epoch, train_loader=train_loader)
            test(test_loader=test_loader)
    
    checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "train_loader": train_loader,
        "test_loader": test_loader
    }

    torch.save(checkpoint, 'model3.pth')

def continue_training(epochs, model_save_file="model3.pth"):
    checkpoint = torch.load(model_save_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']

    train_loader = checkpoint["train_loader"]
    test_loader = checkpoint["test_loader"]

    if __name__ == "__main__":
        for epoch in range(1, epochs + 1):
            current_epoch += 1
            train(current_epoch, train_loader=train_loader)
            test(test_loader=test_loader)
        
    checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "train_loader": train_loader,
        "test_loader": test_loader
    }

    torch.save(checkpoint, model_save_file)

# torch.manual_seed(42)

first_training(10)

# torch.onnx.export(model, torch.randn(64, 3, 64, 64).to(device), "model.onnx", export_params=True)