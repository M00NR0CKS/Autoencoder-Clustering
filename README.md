# Autoencoder-Clustering
K-means++ algae image clustering using autoencoder as a feature extractor. 

This code is part of the module MTHM602 Trends in Data Science and AI: "Computer vision for harmful microalgae identification" project
Method:Autoencoder and K-mean++ Clustering

The autoencoder model for image reconstruction was defined using Pytorch. Convolutional neural network (CNN) architecture for the autoencoder model consists of an encoder that maps input images to a lower-dimensional feature representation and a decoder that maps the feature representation back to the original image space. The encoder consists of multiple convolutional layers followed by ReLU activation functions and max pooling operations, while the decoder also consists of multiple transposed convolutional layers followed by ReLU activation functions and upsampling operations. The mean squared error (MSE) loss is used as the reconstruction loss for training the autoencoder model. All 6,300 images were used for the training operation. The class labels from the image file were extracted for the ground truth labels in the clustering step.

The features extracted from the autoencoder were reshaped and embedded using the t-distributed stochastic neighbour embedding (t-SNE) method with two components. The K-means++ clustering algorithm was performed using the t-SNE features and the elbow plot was used to determine the appropriate number of clusters.The clustering was validated by K-fold validation with 5 folds.Finally, the K-means++ cluster was compared to the true label and evaluated by the average silhouette score.

