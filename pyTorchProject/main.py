import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


# Step 1: Load and normalize the CIFAR10 training and test datasets using torchvision
def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


# Step 2: Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Step 3: Define a loss function
def define_loss():
    return nn.CrossEntropyLoss()


# Step 4: Define the optimizer
def define_optimizer(net, learning_rate=0.001, momentum=0.9):
    return optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


# Step 5: Train the network on the training data
def train_network(net, trainloader, criterion, optimizer, num_epochs=2):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0  # Initialize running loss for each epoch

        # Iterate over the data loader
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


# Step 6: Test the network on the test data
def test_network(net, testloader, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


# Step 7: Save the model trained
def save_model(net, path='./cifar_net.pth'):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the state dictionary of the trained model
    torch.save(net.state_dict(), path)

    print(f"Trained model saved at {path}")


# Utility function to load back the model if exist
def load_model(net, path='./cifar_net.pth'):
    if os.path.exists(path):
        # Load the state dictionary of the trained model
        net.load_state_dict(torch.load(path))
        print(f"Trained model loaded from {path}")
    else:
        print(f"No model found at {path}")


# Utility function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Utility function to test random images
def test_random_images(net, testloader, classes, num_images=5):
    dataiter = iter(testloader)
    for i in range(num_images):
        images, labels = next(dataiter)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(len(images))))
        imshow(torchvision.utils.make_grid(images))


# Helper function to better readability
def print_separator(process_name):
    separator_length = 50
    separator = f"{'=' * separator_length}"
    print(f"\n{separator}\n{process_name}\n{separator}\n")


def main():
    # Load datasets
    print_separator('Loading data...')
    trainloader, testloader, classes = load_datasets()

    # Define the neural network
    print_separator('Building model...')
    net = Net()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    print_separator('Training the network...')
    train_network(net, trainloader, criterion, optimizer)

    # Test the network
    print_separator('Testing the network...')
    test_network(net, testloader, classes)

    # Test random images
    print_separator('Testing random images...')

    #test_random_images(net, testloader, classes)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Save the model
    print_separator('Saving the model...')
    save_model(net)


if __name__ == "__main__":
    main()
