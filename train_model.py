import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class MyDigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 128)  # First layer
        self.layer2 = nn.Linear(128, 64)       # Second layer
        self.layer3 = nn.Linear(64, 10)        # Output layer (0-9)
    
    def forward(self, x):
        #flatten the image so it becomes one long row of pixels
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# normalize the images so numbers are between -1 and 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Downloading the MNIST dataset (handwritten numbers)
train_dataset = datasets.MNIST(
    root="./data",
    train=True, 
    download=True,
    transform=transform
)

# Making batches so it doesnt use too much memory
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Create model
my_model = MyDigitClassifier()

loss_function = nn.CrossEntropyLoss()

# Using Adam optimizer (a popular choice for training neural networks)
optimizer = optim.Adam(my_model.parameters(), lr=0.001)

# Training for 5 rounds (epochs)
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    # Looping through all the images
    for images, labels in train_loader:
        # Clear previous gradients
        optimizer.zero_grad()
        # Make predictions
        predictions = my_model(images)
        # Calculate how wrong we were
        loss = loss_function(predictions, labels)
        # Backpropagation (backpropagation is a fancy word for "figuring out how to change the weights to be less wrong")
        loss.backward()
        # Update weights
        optimizer.step()
        # Add up all the losses
        total_loss = total_loss + loss.item()
    
    # Print average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}") # i have no idea what epoch is btw. i watched a few videos that used it

# Save trained model so it can be used again
torch.save(my_model.state_dict(), "ocr_model.pth")
print("Model saved as ocr_model.pth")