import torch
torch.backends.cudnn.benchmark = True

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DynamicConvNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)

# Datasets and Training Components

# Define transformaitions and datasets
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), # Converts image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalises tensor images (RGB)
]) # Pipeline of transformations applied to every image in dataset

# CIFAR-10 dataset
# Downloads CIFAR-10 training dataset, applies transformation pipeline to each loaded image
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#Loads data in mini-batches of 32
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

# Specifies loading the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# Manages how the CIFAR-10 test dataset is loaded during evaluation phase
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Initialises model class with set parameters
model = DynamicConvNet(num_blocks=4, num_kernels=5, num_classes=10).to(device)

# Training The Model
num_epochs = 150 # Number of times the entire training dataset will be passed through the model

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2) # Loss function (cross-entropy loss)
optimiser = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4) # Adapts learning rate
scheduler = CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=1e-6)



train_losses = []
test_accuracies = []
train_accuracies = []

# Loop running the epochs to train the model
for epoch in range(num_epochs):

  model.train() # Puts model in training mode
  running_loss = 0.0 # Initialises variable to track cumulative loss for the current epoch
  correct = 0 # Initialises variable to count number of correct predictions for the current epoch
  total = 0 # Initalises variable to count total number of examples processed furing the current epoch

# Loop through training dataset in mini-batches (trainloader), inputs are the input images, labels are corresponding ground truth labels
  for inputs, labels in trainloader:

    inputs, labels = inputs.to(device), labels.to(device)
    optimiser.zero_grad() # Clears gradients of all optimised tensors

    outputs = model(inputs.to(device)) # Passes input images through model to get logits

    loss = criterion(outputs, labels.to(device)) # Calculates loss - compares predicted outputs with true labels using CrossEntropyLoss

    loss.backward() # Computes gradients of loss with respect to models parameters (backpropagation)

    optimiser.step() # Updates models parameters based on the calculated gradients and optimisation algorithm (Adam)

    running_loss += loss.item() # Adds current batch's loss to running total

    _, predicted = torch.max(outputs, 1) # Gets predicted class labels - selects class with highest predicted score from models logits
    total += labels.size(0) # Updates total number of samples processed
    correct += (predicted == labels).sum().item() # Counts predictions matching true labels

    current_lr = optimiser.param_groups[0]['lr']

# Output current epochs progress, includes average loss and accuracy.
  scheduler.step()
  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%, Learning Rate: {current_lr:.6f}')

  train_accuracy = 100 * correct / total
  train_accuracies.append(train_accuracy)
  train_losses.append(running_loss / len(trainloader))
# Model Evaluation

  model.eval() # Puts model in evaluation mode
  correct = 0 # Initialises variable to track number of correctly predicted labels
  total = 0 # Initialises variable to count total number of labels in test dataset

  # Disables gradient computation
  with torch.no_grad():

    # Loop over test data
    for inputs, labels in testloader:

      inputs, labels = inputs.to(device), labels.to(device)

      outputs = model(inputs) # Feeds inputs through model to get logits
      _, predicted = torch.max(outputs, 1) # Computes predicted class for each input - selects class with highest logit value

      total += labels.size(0) # Adds batch size to total count
      correct += (predicted == labels).sum().item() # Compares predicted labels with true labels

    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
  # Calculates and prints accuracy of model - divides correct predictions by total number of labels
print(f'Test Accuracy: {100 * correct / total:.2f}%')
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.plot(test_accuracies, label='Test Accuracy', color='red')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curves")
plt.legend()

plt.tight_layout()
plt.show()
