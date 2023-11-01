import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

def load_point_cloud(dataset_file, index):
    with h5py.File(dataset_file, 'r') as file:
        point_cloud = file['data'][index]  # Shape: (2048, 3)
    return point_cloud

def rotate_point_cloud_y(point_cloud, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                [0, 1, 0],
                                [-np.sin(angle_radians), 0, np.cos(angle_radians)],])
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    return rotated_point_cloud

def plot_point_cloud_with_color(ax, point_cloud, title):
    z_coords = point_cloud[:, 2]
    norm = plt.Normalize(z_coords.min(), z_coords.max())
    colors = plt.cm.viridis(norm(z_coords))

    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colors)
    ax.set_title(title)

def find_knn(point_cloud, k_neighbors):
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nn.fit(point_cloud)
    distances, indices = nn.kneighbors(point_cloud)
    return distances, indices

if __name__ == "__main__":
    # Specify the dataset file and index
    dataset_file = 'M:/Data/modelnet40_ply_hdf5_2048/ply_data_train0.h5'
    index = 391
    k_neighbors = 4  # Number of nearest neighbors to find

    # Load the original point cloud
    original_point_cloud = load_point_cloud(dataset_file, index)

    # Rotate the point cloud around the Y-axis
    rotated_point_cloud = rotate_point_cloud_y(original_point_cloud, 60)  # Rotate by 45 degrees

    # Create a 3D plot for the original and rotated point clouds
    fig = plt.figure()

    # Plot the original point cloud with depth-based colors
    ax1 = fig.add_subplot(121, projection='3d')
    plot_point_cloud_with_color(ax1, original_point_cloud, 'Original Point Cloud')

    # Plot the rotated point cloud with depth-based colors
    ax2 = fig.add_subplot(122, projection='3d')
    plot_point_cloud_with_color(ax2, rotated_point_cloud, 'Rotated Point Cloud')

    # Calculate K-nearest neighbors for original and rotated data
    original_distances, original_indices = find_knn(original_point_cloud, k_neighbors)
    rotated_distances, rotated_indices = find_knn(rotated_point_cloud, k_neighbors)

    # Compare K-nearest neighbors
    neighbor_diff = abs(original_distances - rotated_distances)

    # Print K-nearest neighbor information
    print("Original Data - K-Nearest Neighbors:")
    for i in range(len(original_indices)):
        print(f"Point {i} - Neighbors: {original_indices[i]}, Distances: {original_distances[i]}")

    print("Rotated Data - K-Nearest Neighbors:")
    for i in range(len(rotated_indices)):
        print(f"Point {i} - Neighbors: {rotated_indices[i]}, Distances: {rotated_distances[i]}")

    print("Difference in Neighbors (Original - Rotated):")
    print(neighbor_diff)

    plt.show()
