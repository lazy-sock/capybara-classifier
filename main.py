import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import json
import os
import math
from tqdm import tqdm
import torchvision.transforms.functional as TF
from scipy.ndimage import zoom

class AttentionModule(nn.Module):
    """Attention mechanism for focusing on discriminative features"""
    
    def __init__(self, in_channels, reduction=8):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, channels=2048, reduction=4):
        super(CBAM, self).__init__()
        
        # Channel Attention
        self.channelAttention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatialAttention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channelAttention(x)
        x = x * ca

        sa_input = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        sa = self.spatialAttention(sa_input)
        x = x * sa

        return x
        
        
class FineGrainedBirdCNN(nn.Module):
    """Fine-grained CNN for bird species classification"""
    
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.5):
        super(FineGrainedBirdCNN, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add attention modules
        self.attention = CBAM()#AttentionModule(2048)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature enhancement layers
        self.feature_enhance = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
        )
        
        # Classification head
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply attention mechanisms
        attended_features1 = self.attention(features)
        attended_features2 = self.attention(attended_features1)
        attended_features3 = self.attention(attended_features2)
        
        # Global average pooling
        pooled_features = self.global_avg_pool(attended_features3)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Feature enhancement
        enhanced_features = self.feature_enhance(pooled_features)
        
        # Classification
        output = self.classifier(enhanced_features)
        
        return output, attended_features3, enhanced_features  # Return features for visualization

class BirdClassificationTrainer:
    """Trainer class for fine-grained bird classification with ImageFolder support"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.class_names = []
    
    def get_data_transforms(self, input_size=224):
        """Get data augmentation transforms for training and validation"""
        
        train_transforms = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def prepare_data(self, data_dir, train_split=0.7, val_split=0.2, test_split=0.1, 
                     batch_size=32, num_workers=4, random_seed=42):
        """
        Prepare ImageFolder dataset with random splits
        
        Args:
            data_dir: Path to directory containing subdirectories for each class
            train_split: Proportion of data for training
            val_split: Proportion of data for validation  
            test_split: Proportion of data for testing
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            random_seed: Random seed for reproducible splits
        """
        
        # Set random seed for reproducible splits
        torch.manual_seed(random_seed)
        
        # Get transforms
        train_transforms, val_transforms = self.get_data_transforms()
        
        # Create full dataset with validation transforms first to get class info
        full_dataset = ImageFolder(data_dir, transform=val_transforms)
        
        # Store class names
        self.class_names = full_dataset.classes
        num_classes = len(self.class_names)
        
        print(f"Found {len(full_dataset)} images across {num_classes} classes")
        print(f"Classes: {self.class_names}")
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        # Apply different transforms to training set
        train_dataset.dataset = ImageFolder(data_dir, transform=train_transforms)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, num_classes
    
    def show_sample_images(self, data_loader, num_samples=8):
        """Display sample images from the dataset"""
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Denormalize images for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = images[i] * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            axes[i].imshow(img)
            axes[i].set_title(f'{self.class_names[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output, _, _ = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output1, _, _ = self.model(data)
                output2, _, _ = self.model(data)
                output3, _, _ = self.model(data)

                _, pred1 = torch.max(output1.data, 1)
                _, pred2 = torch.max(output2.data, 1)
                _, pred3 = torch.max(output3.data, 1)
                
            
                
                if(torch.equal(pred1, pred2)):
                    output = output1
                elif(torch.equal(pred1, pred3)):
                    output = output1
                elif(torch.equal(pred3, pred2)):
                    output = output2
                else:
                    output = output1
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-4):
        """Full training loop"""
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with different learning rates for different parts
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() if 'backbone' not in n]
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for pretrained backbone
            {'params': other_params, 'lr': lr}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 30)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': self.class_names
                }, 'best_bird_model.pth')
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
        
        return best_val_acc
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def denormalize_image(self, tensor_image):
        """Denormalize image tensor for display"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Move to CPU if on GPU
        if tensor_image.is_cuda:
            tensor_image = tensor_image.cpu()
        
        # Denormalize
        img = tensor_image * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        return img
    
    def analyze_misclassified_images(self, test_loader, max_images_per_class=10):
        """
        Analyze and display misclassified images from the test set
        Shows images with their transformations as the AI sees them
        """
        print("\nAnalyzing misclassified images...")
        
        self.model.eval()
        misclassified_data = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data_gpu = data.to(self.device)
                target_gpu = target.to(self.device)
                
                output, _, _ = self.model(data_gpu)
                _, predicted = torch.max(output, 1)
                
                # Find misclassified images in this batch
                misclassified_mask = (predicted != target_gpu).cpu()
                
                for i in range(len(data)):
                    if misclassified_mask[i]:
                        # Get prediction probabilities for confidence analysis
                        probs = torch.softmax(output[i], dim=0)
                        confidence = probs.max().item()
                        
                        misclassified_data.append({
                            'image': data[i],  # Transformed image as seen by AI
                            'true_label': target[i].item(),
                            'predicted_label': predicted[i].cpu().item(),
                            'confidence': confidence,
                            'all_probs': probs.cpu().numpy()
                        })
        
        if not misclassified_data:
            print("No misclassified images found! Perfect accuracy on test set.")
            return
        
        print(f"Found {len(misclassified_data)} misclassified images")
        
        # Group misclassified images by true class
        misclassified_by_class = {}
        for item in misclassified_data:
            true_class = item['true_label']
            if true_class not in misclassified_by_class:
                misclassified_by_class[true_class] = []
            misclassified_by_class[true_class].append(item)
        
        # Sort by confidence (lowest first - most uncertain predictions)
        for class_idx in misclassified_by_class:
            misclassified_by_class[class_idx].sort(key=lambda x: x['confidence'])
        
        # Create visualization
        self._plot_misclassified_images(misclassified_by_class, max_images_per_class)
        
        # Print detailed analysis
        self._print_misclassification_analysis(misclassified_by_class)
    
    def _plot_misclassified_images(self, misclassified_by_class, max_images_per_class):
        """Plot misclassified images organized by true class"""
        
        # Calculate grid dimensions
        num_classes_with_errors = len(misclassified_by_class)
        
        for class_idx, misclassified_items in misclassified_by_class.items():
            class_name = self.class_names[class_idx]
            items_to_show = misclassified_items[:max_images_per_class]
            
            if not items_to_show:
                continue
            
            # Calculate subplot dimensions
            n_images = len(items_to_show)
            cols = min(5, n_images)
            rows = math.ceil(n_images / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
            
            # Handle single image case
            if n_images == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if n_images > 1 else [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'Misclassified Images - True Class: {class_name}', 
                        fontsize=16, fontweight='bold')
            
            for i, item in enumerate(items_to_show):
                if i >= len(axes):
                    break
                
                # Denormalize and display image
                img = self.denormalize_image(item['image'])
                axes[i].imshow(img)
                
                # Create detailed title
                predicted_class = self.class_names[item['predicted_label']]
                confidence = item['confidence']
                
                title = f"Predicted: {predicted_class}\n"
                title += f"Confidence: {confidence:.2f}"
                
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
                
                # Add a red border to emphasize it's misclassified
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            
            # Hide unused subplots
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def _print_misclassification_analysis(self, misclassified_by_class):
        """Print detailed analysis of misclassifications"""
        
        print("\n" + "="*80)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*80)
        
        total_misclassified = sum(len(items) for items in misclassified_by_class.values())
        
        print(f"Total misclassified images: {total_misclassified}")
        print(f"Classes with misclassifications: {len(misclassified_by_class)}")
        
        # Analyze confusion patterns
        confusion_pairs = {}
        confidence_stats = {'low': 0, 'medium': 0, 'high': 0}
        
        for class_idx, items in misclassified_by_class.items():
            true_class = self.class_names[class_idx]
            
            print(f"\n{true_class} (misclassified: {len(items)})")
            print("-" * 50)
            
            # Count predictions for this true class
            pred_counts = {}
            confidences = []
            
            for item in items:
                pred_class = self.class_names[item['predicted_label']]
                pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1
                confidences.append(item['confidence'])
                
                # Track confusion pairs
                pair = (true_class, pred_class)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
                
                # Categorize confidence
                if item['confidence'] < 0.5:
                    confidence_stats['low'] += 1
                elif item['confidence'] < 0.8:
                    confidence_stats['medium'] += 1
                else:
                    confidence_stats['high'] += 1
            
            # Print top confusions for this class
            sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
            for pred_class, count in sorted_preds[:3]:  # Top 3 confused classes
                print(f"  → Often predicted as: {pred_class} ({count} times)")
            
            # Print confidence statistics for this class
            avg_conf = np.mean(confidences)
            print(f"  → Average wrong confidence: {avg_conf:.3f}")
        
        # Overall confusion analysis
        print(f"\n{'='*50}")
        print("CONFIDENCE ANALYSIS")
        print(f"{'='*50}")
        print(f"Low confidence errors (<0.5): {confidence_stats['low']}")
        print(f"Medium confidence errors (0.5-0.8): {confidence_stats['medium']}")
        print(f"High confidence errors (>0.8): {confidence_stats['high']}")
        
        # Top confusion pairs
        print(f"\n{'='*50}")
        print("TOP CONFUSION PAIRS")
        print(f"{'='*50}")
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        for i, ((true_class, pred_class), count) in enumerate(sorted_pairs[:5]):
            print(f"{i+1}. {true_class} → {pred_class}: {count} times")
    
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output, _, _ = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.numpy())
        
        # Overall accuracy
        accuracy = 100 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print("=" * 50)
        report = classification_report(all_targets, all_predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(max(8, len(self.class_names) * 0.6), max(6, len(self.class_names) * 0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return report
    
    def load_model(self, checkpoint_path):
        """Load a saved model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint['class_names']
        print(f"Model loaded from {checkpoint_path}")
        print(f"Best validation accuracy was: {checkpoint['best_val_acc']:.2f}%")

    def show_heatmap_on_image(self, train_loader):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        images_per_window = 36
        collected_imgs = []
        collected_heatmaps = []

        for batch_id, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                _, heatmaps, _ = self.model(data)

            for b in range(data.size(0)):
                img_tensor = data[b].cpu()
                heatmap_tensor = heatmaps[b].cpu()

                # Denormalisieren und konvertieren
                img = img_tensor * std + mean
                img = np.array(TF.to_pil_image(img).convert("RGB"))

                # Heatmaps summieren
                heatmap_sum = np.zeros_like(heatmap_tensor[0])
                for pHeatmap in heatmap_tensor:
                    heatmap_sum += pHeatmap.numpy()

                # Heatmap auf Bildgröße skalieren
                heatmap_resized = zoom(
                    heatmap_sum,
                    (img.shape[0] / heatmap_sum.shape[0], img.shape[1] / heatmap_sum.shape[1]),
                    order=3
                )

                collected_imgs.append(img)
                collected_heatmaps.append(heatmap_resized)

                # Wenn genug Bilder gesammelt, anzeigen
                if len(collected_imgs) == images_per_window:
                    fig, axes = plt.subplots(4, math.ceil(images_per_window/4), figsize=(17, 7))
                    if isinstance(axes, np.ndarray):
                        axes = axes.flatten()
                    for i in range(images_per_window):
                        axes[i].imshow(collected_imgs[i])
                        axes[i].imshow(collected_heatmaps[i], cmap='jet', alpha=0.3)
                        axes[i].axis('off')
                    
                    for j in range(images_per_window, len(axes)):
                        axes[j].axis('off')
                    plt.tight_layout()
                    plt.show()

                    # Leeren
                    collected_imgs.clear()
                    collected_heatmaps.clear()

                    # Benutzerabfrage
                    if input("Weitere Heatmaps anzeigen? (y/n): ").lower() != 'y':
                        return

        # Noch verbleibende Bilder anzeigen
        if collected_imgs:
            fig, axes = plt.subplots(1, len(collected_imgs), figsize=(15, 5))
            for i in range(len(collected_imgs)):
                axes[i].imshow(collected_imgs[i])
                axes[i].imshow(collected_heatmaps[i], cmap='jet', alpha=0.3)
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
                               
def main_training_pipeline(data_dir, batch_size, epochs, lr, grid_search):
    """
    Complete training pipeline for bird classification
    
    Args:
        data_dir: Path to directory containing bird images organized in subdirectories
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        grid_search: If True, runs a grid search without visualizations
    """
    
    print("Starting Bird Classification Training Pipeline")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        return None, None, None, None
    
    # Initialize model (we'll set num_classes after loading data)
    model = None
    trainer = None
    
    #try:
    # Prepare data
    temp_trainer = BirdClassificationTrainer(FineGrainedBirdCNN(10))  # Temporary
    train_loader, val_loader, test_loader, num_classes = temp_trainer.prepare_data(
        data_dir, batch_size=batch_size
    )
    
    # Create the actual model with correct number of classes
    model = FineGrainedBirdCNN(num_classes=num_classes)
    trainer = BirdClassificationTrainer(model)
    trainer.class_names = temp_trainer.class_names    
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Show sample images
    if not grid_search:
        print("\nSample images from dataset:")
        trainer.show_sample_images(train_loader)
    
    # Train the model
    print("\nStarting training...")
    accuracy = trainer.train(train_loader, val_loader, epochs=epochs, lr=lr)
    
    # Plot training history
    if not grid_search:
        trainer.plot_training_history()
    
    # Evaluate on test set
    if not grid_search:
        print("\nEvaluating on test set...")
        trainer.evaluate_model(test_loader)
    
    # Analyze misclassified images
    if not grid_search:
        print("\nAnalyzing misclassified images...")
        trainer.analyze_misclassified_images(test_loader, max_images_per_class=8)
        
    trainer.show_heatmap_on_image(train_loader)
    
    return trainer, train_loader, val_loader, test_loader, accuracy
    
        
        
    #except Exception as e:
    #    print(f"Error during training: {str(e)}")
    #    return None, None, None, None, None

def gridsearch(data_directory):
    param_grid = {
        'batch_size': [16, 32],
        'epochs': [6, 12],
        'lr': [0.001, 0.0001]
    }
    
    from itertools import product
    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))
    results = []
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        print("=" * 50)
        print(f"Testing combination: {params}")
        
        try:
            trainer, train_loader, val_loader, test_loader, accuracy = main_training_pipeline(
                data_dir = data_directory,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                lr=params['lr'],
                grid_search=True
            )
            results.append((params, accuracy))
            
        except Exception as e:
            print(f"Error with combination {params}: {str(e)}")
        
    best_params = max(results, key=lambda x: x[1])
    print("Best parameters found:" , best_params[0])
    print("Best accuracy:", best_params[1])

if __name__ == "__main__":
    data_directory = "d:\Code\BWKI\capybara-classifier\images/bird_dataset_v3/birds"  # Change this to your dataset path
    
    #gridsearch(data_directory)
    
    
    trainer, train_loader, val_loader, test_loader, accuracy = main_training_pipeline(
        data_dir=data_directory,
        batch_size=32,
        epochs=15,
        lr=0.001,
        grid_search=False
    )
    
    
    