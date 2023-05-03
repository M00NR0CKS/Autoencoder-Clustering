# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:31:54 2023

@author: tp504
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score
from mpl_toolkits.mplot3d import Axes3D

#%% Define Custom Dataset 
# Define the custom dataset class for loading algae data from drive 
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []  # List to store image paths

        # Load image file paths
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg"):  # images are in jpg format
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


#%% Define data directory from drive
# Define the data directory on drive 
data_dir = 'D:\Data_AlgaeImage'  

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256
    transforms.CenterCrop(256),  # Crop images to 256x256 at the center
    transforms.ToTensor(),  # Convert images to tensors
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize image tensors
])

# Create an instance of the custom dataset
dataset = MyDataset(data_dir, transform=transform)

# Create a data loader to load data in batches
batch_size = 64  
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#%% Define Autoencoder
# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Tanh()  # use Tanh activation to ensure output is in range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create an instance of the autoencoder model
autoencoder = Autoencoder()
#%%
# Resize the input tensor to match the size of the target tensor
def resize_input(input_tensor, target_size):
    return F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)

# Calculate the MSE loss with resized input and target tensors
criterion = nn.MSELoss()
#%%
# Define the optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
# Define scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#%%
# Create an instance of the custom dataset
dataset = MyDataset(data_dir, transform=transform)

# Create a data loader to load data in batches
batch_size = 64  
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#%%Train the model



# Training loop
num_epochs = 10  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
num_classes = 12
autoencoder.to(device)

# Lists to store training progress
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    for batch_idx, images in enumerate(data_loader):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(images)
        inputs_resized = resize_input(outputs, images.size()[2:])
        loss = criterion(inputs_resized, images)
        loss.backward()
        optimizer.step()
        
        # Print loss
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")
            
        # Store loss for visualization
        train_losses.append(loss.item())

    # Update learning rate
    scheduler.step()

print("Finished training")
# Save the trained model
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
#%%
autoencoder = Autoencoder()  
autoencoder.load_state_dict(torch.load('autoencoder.pth'))  # Load trained weights
autoencoder.eval()  # Set to evaluation mode
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
encoder = autoencoder.encoder.to(device)  # Extract encoder from autoencoder and move to device

# Loop through the dataset and extract features for each image
features = []
for batch_idx, images in enumerate(data_loader):
    images = images.to(device)
    encoded = encoder(images)  # Encode images to get features
    features.append(encoded.detach().cpu().numpy())  # Convert to numpy array and store

# Concatenate features from all batches
features = np.concatenate(features, axis=0)

#%% Assign true labels
tlabel_df = pd.DataFrame(columns=["img_name","label"])
tlabel_df["img_name"] = os.listdir("D:\Data_AlgaeImage")
for idx, i in enumerate(os.listdir("D:\Data_AlgaeImage")):
    # print(i)
    if "Anabaena" in i:
        tlabel_df["label"][idx] = 0
        # print(train_df)
    if "Aphanizomenon" in i:
        tlabel_df["label"][idx] = 1
    if "Gymnodinium" in i:
        tlabel_df["label"][idx] = 2
    if "Karenia" in i:
        tlabel_df["label"][idx] = 3
    if "Microcystis" in i:
        tlabel_df["label"][idx] = 4
    if "Noctiluca" in i:
        tlabel_df["label"][idx] = 5
    if "Nodularia" in i:
        tlabel_df["label"][idx] = 6
    if "nontoxic" in i:
        tlabel_df["label"][idx] = 7
    if "Nostoc" in i:
        tlabel_df["label"][idx] = 8
    if "Oscillatoria" in i:
        tlabel_df["label"][idx] = 9
    if "Prorocentrum" in i:
        tlabel_df["label"][idx] = 10
    if "Skeletonema" in i:
        tlabel_df["label"][idx] = 11                   
print(tlabel_df)
tlabel_df.to_csv (r'annotation.csv', index = False, header=True)
#%%
# Load ground truth labels from annotated file
# load the labels and store them in a numpy array y_true with shape (n_samples,)
# Read the annotations.csv file using pandas
df = pd.read_csv('annotation.csv')

# Extract the "label" column as a numpy array
y_true = df['label'].values

# Convert the numpy array to int32 data type
y_true = y_true.astype(np.int32)
#%%
# Load and preprocess image data 
# The image have already loaded and preprocessed, and stored it in a numpy array X with shape (n_samples, n_features)
n_samples, width, height, channels = features.shape
X = features.reshape(n_samples, width * height * channels)

# Perform dimensionality reduction 
tsne = TSNE(n_components=2, random_state = 60)
X_tsne = tsne.fit_transform(X)

# Save the t-SNE results to a file
np.save('tsne_results.npy', X_tsne)

# Define label mapping
label_mapping = {
    0: 'Anabaena',
    1: 'Aphanizomenon',
    2: 'Gymnodinium',
    3: 'Karenia',
    4: 'Microcystis',
    5: 'Noctiluca',
    6: 'Nodularia',
    7: 'nontoxic',
    8: 'Nostoc',
    9: 'Oscillatoria',
    10: 'Prorocentrum',
    11: 'Skeletonema'
}

# Load true labels
df = pd.read_csv('annotation.csv')
y_true = df['label'].values
y_true = y_true.astype(np.int32)

# Map labels to new labels
y_true_mapped = np.array([label_mapping[label] for label in y_true])

# Perform k-fold cross-validation with K-means++ 
k = 12 
n_splits = 5 
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=60)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
accuracy_scores = []
f1_scores = []


for train_index, test_index in kf.split(X_tsne):
    X_train, X_test = X_tsne[train_index], X_tsne[test_index]
    y_train, y_test = y_true[train_index], y_true[test_index]
    
    kmeans.fit(X_train)
    labels = kmeans.predict(X_test)
    
    # Map predicted labels to new labels
    y_pred = np.array([label_mapping[label] for label in labels])
    
    accuracy = accuracy_score(y_test, labels)
    f1 = f1_score(y_test, labels, average='weighted')
    
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    
    print('Accuracy Score (Fold {}): {:.2f}'.format(n_splits + 1, accuracy))
    print('F1 Score (Fold {}): {:.2f}'.format(n_splits + 1, f1))


# Compute average accuracy and F1 score 
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)

# Print average accuracy and F1 score
print('Average Accuracy Score: {:.2f}'.format(avg_accuracy))
print('Average F1 Score: {:.2f}'.format(avg_f1))

#%%
# Map labels to colors
label_color_mapping = {
    'Anabaena': 'red',
    'Aphanizomenon': 'blue',
    'Gymnodinium': 'green',
    'Karenia': 'orange',
    'Microcystis': 'purple',
    'Noctiluca': 'brown',
    'Nodularia': 'gray',
    'nontoxic': 'black',
    'Nostoc': 'pink',
    'Oscillatoria': 'yellow',
    'Prorocentrum': 'magenta',
    'Skeletonema': 'cyan'
}
#%%

# Map true labels to new labels
y_true_mapped = np.array([label_mapping[label] for label in y_true])

# Fit K-means on the entire dataset
kmeans.fit(X_tsne)
kmeans_labels = kmeans.labels_

# Map K-means labels to new labels
kmeans_labels_mapped = np.array([label_mapping[label] for label in kmeans_labels])
#kmeans_labels_color = [label_color_mapping[label] for label in kmeans_labels_mapped]
#y_true_color = [label_color_mapping[label] for label in y_true_mapped]

#%%
for label in label_mapping.values():
    plt.scatter(X_tsne[kmeans_labels_mapped==label, 0], X_tsne[kmeans_labels_mapped==label, 1],cmap='viridis', alpha=1, label=label, s=20)
    #plt.scatter(X_tsne[y_true_mapped==label, 0], X_tsne[y_true_mapped==label, 1], c=label_color_mapping[label], marker='x', label=label + ' (True)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='maroon', marker='x', s=100)
plt.title('K-means++ Clustering Results')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')    
plt.legend(bbox_to_anchor=(1, 1), fontsize = 8)
plt.show()

#%% Plot the true label
for label in label_mapping.values():
    #plt.scatter(X_tsne[kmeans_labels_mapped==label, 0], X_tsne[kmeans_labels_mapped==label, 1], c=label_color_mapping[label], alpha=0.3, label=label + ' (K-means)')
    plt.scatter(X_tsne[y_true_mapped==label, 0], X_tsne[y_true_mapped==label, 1],cmap='viridis', marker='x', label=label)
plt.title('True Label Resutls')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1, 1),fontsize = 8)
plt.show()

#%%
silhouette_avg = silhouette_score(X_test, labels)
print("The average silhouette_score is :", silhouette_avg)