import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load phase data from JSON file
with open('phase_response.json', 'r') as f:
    phase_data = json.load(f)

# Load amplitude data from JSON file
with open('amplitude_response.json', 'r') as f:
    amplitude_data = json.load(f)

# Convert phase data from degrees to radians
for key in phase_data:
    for freq in phase_data[key]:
        phase_data[key][freq] = np.deg2rad(phase_data[key][freq])

# Define unit keys (1 to 16 as strings)
unit_keys = [str(i) for i in range(1, 17)]

# Transform phase data into a tensor of shape (16, frequency_count)
units = torch.tensor([[phase_data[key][str(freq)] for freq in sorted(phase_data[key].keys(), key=int)] 
                      for key in unit_keys], dtype=torch.float32).to(device)

# Transform amplitude data into a tensor of shape (16, frequency_count)
amplitude_data = torch.tensor([[amplitude_data[key][str(freq)] for freq in sorted(amplitude_data[key].keys(), key=int)] 
                      for key in unit_keys], dtype=torch.float32).to(device)

print(units.shape)  # Shape: (16, frequency_count)

# Function to compute phase response for a given unit, position, and set of directions
def compute_phase_response(unit, position, theta, freqs, d=0.01, c=343):
    """
    Compute the phase response for a given unit and its position.

    Parameters:
    - unit: the unit index (1 to 16)
    - position: the position index (1 to num_units)
    - theta: torch tensor of angles (in degrees)
    - freqs: torch tensor of frequencies
    - d: distance increment between positions
    - c: speed of sound (343 m/s)

    Returns:
    - magnitude_response: complex tensor of shape (len(theta), len(freqs))
    """
    magnitude_response = torch.zeros(len(theta), len(freqs), dtype=torch.cfloat, device=device)
    for i, angle in enumerate(theta):
        for j, freq in enumerate(freqs):
            angle_rad = torch.deg2rad(torch.tensor(angle, device=device))
            phase_shift = units[unit - 1][j] + 2 * np.pi * freq * ((position - 1) * d * torch.sin(angle_rad)) / c
            magnitude_response[i, j] += torch.exp(1j * phase_shift) * amplitude_data[unit - 1][j]
            magnitude_response[i, j] *= torch.cos(angle_rad)
    return magnitude_response

# Function to compute cosine similarity between two responses
def cosine_similarity(response1, response2):
    """
    Compute the cosine similarity between two responses.

    Parameters:
    - response1, response2: response vectors of the same shape

    Returns:
    - Average cosine similarity
    """
    return F.cosine_similarity(response1, response2, dim=0).mean()

# Loss function definition
def loss_function(similarity_matrix, G_max, beta):
    """
    Compute the loss function based on the similarity matrix and G_max.

    Parameters:
    - similarity_matrix: computed similarity matrix
    - G_max: maximum similarity value in the matrix
    - beta: weight for the G_max penalty term

    Returns:
    - Loss value
    """
    return torch.sum(similarity_matrix) + beta * G_max

# Function to compute similarity matrix
def compute_similarity_matrix(ams_configuration, magnitude_responses):
    """
    Compute the similarity matrix for all directions.

    Parameters:
    - ams_configuration: AMS configuration tensor
    - magnitude_responses: tensor of magnitude responses for all units and frequencies

    Returns:
    - similarity_matrix: a symmetric matrix of similarities between directions
    """
    num_directions = magnitude_responses.shape[2]
    similarity_matrix = torch.zeros((num_directions, num_directions), device=device)
    for i in range(num_directions):
        for j in range(num_directions):
            if i != j:
                response1 = magnitude_responses[:, :, i, :].sum(dim=1)
                response2 = magnitude_responses[:, :, j, :].sum(dim=1)
                similarity_matrix[i, j] = cosine_similarity(response1, response2)
    return similarity_matrix

# Generate angles (-45° to 45°) and frequencies
theta = torch.linspace(-45, 45, 90).to(device)
freqs = torch.tensor([16000 + i * 100 for i in range(41)], device=device)

# Initialize AMS configuration proxy variable
num_units = 100  # Number of units
ams_config_proxy = torch.randn((num_units, 16), requires_grad=True, device=device)

# Define optimizer
optimizer = Adam([ams_config_proxy], lr=0.2)
beta = 0.2  # Regularization weight for G_max

# Compute magnitude responses for all units and positions
magnitude_responses_list = [[compute_phase_response(unit, k, theta, freqs) 
                              for k in range(1, 1 + num_units)] for unit in range(1, 17)]
magnitude_responses_tensor = torch.stack([torch.stack(inner_list) for inner_list in magnitude_responses_list]).to(device)

# Function to discretize AMS configuration
def get_discrete_configuration(ams_configuration):
    """
    Convert AMS configuration probabilities into discrete states.
    """
    discrete_configuration = torch.zeros_like(ams_configuration)
    max_indices = torch.argmax(ams_configuration, dim=1)
    discrete_configuration[torch.arange(ams_configuration.size(0)), max_indices] = 1
    return discrete_configuration

# Compute the initial similarity matrix
initial_ams_configuration = F.softmax(ams_config_proxy, dim=1)
initial_discrete_configuration = get_discrete_configuration(initial_ams_configuration)
initial_similarity_matrix = compute_similarity_matrix(initial_discrete_configuration, magnitude_responses_tensor)

# Optimization loop
num_iterations = 100
for iteration in range(num_iterations):
    optimizer.zero_grad()
    ams_configuration = F.softmax(ams_config_proxy, dim=1)
    similarity_matrix = compute_similarity_matrix(ams_configuration, magnitude_responses_tensor)
    G_max = torch.max(similarity_matrix)
    loss = loss_function(similarity_matrix, G_max, beta)
    loss.backward()
    optimizer.step()
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")

# Compute final similarity matrix
final_ams_configuration = F.softmax(ams_config_proxy, dim=1)
final_discrete_configuration = get_discrete_configuration(final_ams_configuration)
final_similarity_matrix = compute_similarity_matrix(final_discrete_configuration, magnitude_responses_tensor)

# Save and display results
ams_configuration_final = final_discrete_configuration.detach().cpu().numpy()
ams_selected_states = np.argmax(ams_configuration_final, axis=1)
print("Final AMS configuration (selected states for each unit):")
print(ams_selected_states)

# Function to plot similarity matrix
def plot_similarity_matrix(matrix, title):
    """
    Plot a similarity matrix as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.title(title, fontsize=16)
    plt.xlabel('Direction Index', fontsize=14)
    plt.ylabel('Direction Index', fontsize=14)
    plt.show()

# Save matrices and plot results
np.savetxt("initial_similarity_matrix.txt", initial_similarity_matrix.cpu().numpy(), fmt="%.6f")
np.savetxt("final_similarity_matrix.txt", final_similarity_matrix.cpu().numpy(), fmt="%.6f")
plot_similarity_matrix(initial_similarity_matrix.cpu().numpy(), 'Initial Similarity Matrix')
plot_similarity_matrix(final_similarity_matrix.cpu().numpy(), 'Final Similarity Matrix')
