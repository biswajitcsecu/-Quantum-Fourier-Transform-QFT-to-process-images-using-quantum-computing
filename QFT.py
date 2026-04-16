import cv2
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import QFT

def preprocess_image(image_path, size=(4, 4)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

def encode_image(img):
    flattened_img = img.flatten()
    norm_factor = np.linalg.norm(flattened_img)
    if norm_factor == 0:
        raise ValueError("Image normalization factor is zero.")
    quantum_state = flattened_img / norm_factor
    return quantum_state

def apply_qft(quantum_state):
    num_qubits = int(np.log2(len(quantum_state)))
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.append(QFT(num_qubits), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

def measure_and_reconstruct(qc, backend, size):
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=1024)
    result = job.result()
    
    if not result.results:
        print("Warning: No measurement results found. Ensure correct backend and shots are used.")
        return np.zeros(size)
    
    counts = result.get_counts() if hasattr(result, 'get_counts') else {}
    print("First 10 Measurement Results:", list(counts.items())[:10])
    
    img_reconstructed = np.zeros(size)

    if not counts:
        print("Warning: No quantum measurement data available.")
        return img_reconstructed

    min_count = min(counts.values())
    max_count = max(counts.values())
    print(f"Min Count: {min_count}, Max Count: {max_count}")

    for key, count in counts.items():
        index = int(key, 2) % (size[0] * size[1])
        row, col = divmod(index, size[1])
        img_reconstructed[row, col] = np.interp(count, [min_count, max_count], [0, 255])

    return img_reconstructed.astype(np.uint8)

QiskitRuntimeService.save_account("apikey", overwrite=True, channel="ibm_quantum")
service = QiskitRuntimeService()

try:
    backend = service.get_backend('ibm_kyiv')
except:
    backend = Aer.get_backend('aer_simulator')

image_path = "doge.jpg"
size = (1024, 1024)
img = preprocess_image(image_path, size)
print("Processed Image Shape:", img.shape)

quantum_state = encode_image(img)
qc = apply_qft(quantum_state)
img_reconstructed = measure_and_reconstruct(qc, backend, size)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(img_reconstructed, cmap="gray")
plt.title("Quantum Processed Image")
plt.show()