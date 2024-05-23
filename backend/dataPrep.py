import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split


#define the neural network
class MusicGenreClassifier(nn.Module):
    def __init__(self):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 4 input features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(unique_labels))  # number of output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#open files with read
with open('features.json', 'r') as f:
    features_list = json.load(f)

with open('labels.json', 'r') as f:
    labels_list = json.load(f)


#extract specific features using numpy ndarray
#ex output array([
#    [0.8, 0.6, 120.0, 0.5],
#    [0.7, 0.8, 130.0, 0.6],
#    [0.6, 0.9, 140.0, 0.7]
#])
#each row represents an individual song, columns represent the features
X = np.array([[f['danceability'], f['energy'], f['tempo']], f['valence'] for f in features_list])


#encode labels as integers
#create the mapping from label to integer and #encode the labels using the created mapping
unique_labels = list(set(labels_list))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_int[label] for label in labels_list])

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating a custom dataset using torches Dataset primitive
class CustomMusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    #number of samples
    def __len__(self):
        return len(self.features)
    
    #returns a sample at a given index
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Create dataset and dataloader instances
train_dataset = CustomMusicDataset(X_train, y_train)
test_dataset = CustomMusicDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#initialize model, loss function, and optimizer
model = MusicGenreClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

#eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')