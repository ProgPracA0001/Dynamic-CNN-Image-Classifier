import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialise Stem class
class Stem(nn.Module):

  def __init__(self):
    # Create a convolutional layer
    super(Stem, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)

    # Create max pooling layer
    self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)



# Define forward pass
  def forward(self, x):

    x = self.conv1(x) # Convolutional layer to extract features
    x = F.relu(x) # Apply non-liearity
    x = self.pool(x) # Down-sample using max pooling

    return x

# Define Expert Branch
class ExpertBranch(nn.Module):

  # Expert Branch Constructor

  # in_channels: Number of input chanels from the previous layer
  # reduction_ratio: Used to shrink the number of channels in the intermediate layer for efficiency
  # num_kernels: How many different convolutional branches/kernels we're learning to to weight

  def __init__(self, in_channels, reduction_ratio=2, num_kernels=5):

    super(ExpertBranch, self).__init__() # Calls the parent nn.Module constructor

    reduced_channels = in_channels // reduction_ratio # Reduces number of channels in the middle layer

    self.pool = nn.AdaptiveAvgPool2d(1) # Applies global average pooling to each feature map

    self.fc1 = nn.Linear(in_channels, reduced_channels) # First Fully Connected layer
    self.fc2 = nn.Linear(reduced_channels, num_kernels) # Second Fulle Connected layer


# Define the forward pass (how data flows through layers)
  def forward(self, x):

    batch_size , channels, _, _ = x.size() # Reads input tensor shape

    x = self.pool(x).view(batch_size, channels) # Apple global avg pooling over spatial dimensions

    x = F.relu(self.fc1(x)) # Transform input into a compact latent representation (non-linear activation)
    x = F.softmax(self.fc2(x), dim=1) # Gives probabiity distribution over the expert branches

    return x # Return final softmaxed vector of attention weights


# Define convolutional branch that contains multiple parallel convolutional layers
class ConvBranch(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size=3, num_kernels=5):

    super(ConvBranch, self).__init__() # Initialises parent class so PyTorch can manage parameters

    # self.convs is a list of convolutional layers
    self.convs = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1), # padding ensures output spatial size is preserved when using kernel_size
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        for _ in range(num_kernels)
    ])

#
  def forward(self, x, weights):
    out = 0 # initialises final output as zero

    # Loop over each convolution layer and its index
    for i, conv in enumerate(self.convs):
      weighted_output = weights[:, i].view(-1, 1, 1, 1) * conv(x)
      out += weighted_output

    return out

# Define new block dynamically combining multiple convolutional outputs based on the learned attention
class DynamicBlock(nn.Module):

  # Initialise block with input, output channels and number of kernels
  def __init__(self, in_channels, out_channels, num_kernels=4):

    super(DynamicBlock, self).__init__() # Calls parent constructor

    # Initialise expert branch
    self.expert = ExpertBranch(in_channels, num_kernels=num_kernels)

    # Initialise convolutional branch
    self.conv_branch = ConvBranch(in_channels, out_channels, num_kernels=num_kernels)

# Define how the input flows through dynamic block
  def forward(self, x):

    # Passes input through expert branch, gets attention weights for each
    weights = self.expert(x)

    # Apply conv branches using learned weights to combine their outputs
    x = self.conv_branch(x, weights)

    return x # Return combined output

# Define classifier module for final imagge classification
class Classifier(nn.Module):

  # Initialise classifier
  def __init__(self, in_features, num_classes):

    super(Classifier, self).__init__() # Fully connected layer that maps extracted features

    self.fc = nn.Linear(in_features, num_classes)

# Define Input flow
  def forward(self, x):

    x = F.adaptive_avg_pool2d(x, 1) # Apply adaptive average pooling to reduce each feature map to a single value (global avg pooling)
    x = x.view(x.size(0), -1) # Flatten pooled feature map to single value
    x = self.fc(x) # Apply fully connected layer to get final class scores

    return x # Return raw logits

# Define full dynamic convolutional neural network model
class DynamicConvNet(nn.Module):

  # Initialise layers of the model
  def __init__(self, num_blocks=4, num_kernels=5, num_classes=10):

    super(DynamicConvNet, self).__init__() # Inherit from nn.Module

    print("Initialising DynamicConvNet model...")

    self.stem = Stem() # Initialise feature extraction layer

    blocks = [] # List to store all dynamic blocks

    in_channels = 64 # Number of channels output from Stem layer

    # Create sequence of dynamic blocks
    for _ in range(num_blocks):
      # Each block takes in parameters
      blocks.append(DynamicBlock(in_channels, in_channels, num_kernels=num_kernels))

    self.backbone = nn.Sequential(*blocks) # Combine list of dynamic blocks into one module

    self.classifier = Classifier(in_channels, num_classes) # Final classification layer to predict the class of the input image

    self.dropout = nn.Dropout(p=0.3)
  # Define how data flows through the full network
  def forward(self, x):

    x = self.stem(x) # Visual features: edges, shapes, textures

    x = self.backbone(x) # Pass through dynamic convolutional layers

    x = self.dropout(x)

    x = self.classifier(x) # VConvert final feature maps to logits

    return x # Return logits