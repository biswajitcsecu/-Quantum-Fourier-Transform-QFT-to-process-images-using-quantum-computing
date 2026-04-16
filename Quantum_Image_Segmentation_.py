#!/usr/bin/env python
# coding: utf-8

# ## **Quantum Medical Image Segmentation**

# In[49]:


from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.filters import sobel
from PIL import Image


# In[50]:


# Quantum device setup
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)


# ## **Initialize Qubits**

# In[51]:


def initializeQubits(data): 
    @qml.qnode(dev)
    def encode(state):
        for i, val in enumerate(state):
            qml.RY(val * np.pi, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(len(state))]

    encoded_states = []
    for point in data:
        normalized = point / np.linalg.norm(point)  
        encoded_states.append(encode(normalized))
    return np.array(encoded_states)


# ## **Quantum Kmeans**

# In[52]:


def hybridKmeans(data, k, max_iters=10):
    classical_kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42).fit(data)
    centroids = classical_kmeans.cluster_centers_
    quantum_centroids = initializeQubits(centroids)

    for iteration in range(max_iters):
        labels = []
        for point in data:
            distances = [np.linalg.norm(point - c) for c in quantum_centroids]
            labels.append(np.argmin(distances))

        labels = np.array(labels)
        unique_labels = len(set(labels))
        if unique_labels <= 1:
            centroids = classical_kmeans.cluster_centers_  
            quantum_centroids = initializeQubits(centroids)
            continue

        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)

        quantum_centroids = initializeQubits(centroids)

    return centroids, labels


# ## **Data Processing**

# In[53]:


def preprocess_image(image_path):   
    image = Image.open(image_path).convert("L")
    image_data = np.array(image) / 255.0 

    # Extract edges features 
    edges = sobel(image_data)
    flat_data = np.stack((image_data.flatten(), edges.flatten()), axis=1)

    # Standardize features for best clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flat_data)
    return image, image_data, scaled_data


# ## ***Display***

# In[54]:


def visualize_results(image, image_data, labels_quantum, labels_classical, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(labels_quantum.reshape(image_data.shape[:2]), cmap='viridis')
    axes[1].set_title("Quantum K-Means Segmentation")
    axes[1].axis('off')

    axes[2].imshow(labels_classical.reshape(image_data.shape[:2]), cmap='viridis')
    axes[2].set_title("Classical K-Means Segmentation")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ## **------------------Main-----------------**

# In[ ]:


img_path = "img1.jpg"
image, image_data, processed_data = preprocess_image(img_path)

# Parameters
k = 4
max_iters = 10

# Apply Hybrid Quantum K-Means
print("Running Hybrid Quantum K-Means...")
quantum_centroids, quantum_labels = hybridKmeans(processed_data, k, max_iters)

# Apply Classical K-Means
print("Running Classical K-Means...")
classical_kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42).fit(processed_data)
classical_labels = classical_kmeans.labels_

# Reshape labels back to image 
quantum_segmented = quantum_labels.reshape(image_data.shape)
classical_segmented = classical_labels.reshape(image_data.shape)

# Visualize and save results
output_image = "quantum_segmentation.png"
visualize_results(image,image_data, quantum_segmented, classical_segmented, output_image)
print(f"Segmentation comparison saved to {output_image}.")


# In[ ]:




