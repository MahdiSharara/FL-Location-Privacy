import numpy as np
from typing import Dict
from data_structures import User, Node, Link
from utilities.load_json import load_config
import logging
import importlib
import generate_data
importlib.reload(generate_data)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for configuration keys
CONFIG_KEYS = {
    "x_y_location_range": "x_y_location_range",
    "z_location_range": "z_location_range",
    "rate_requirement": "rate_requirement",
    "delay_requirement": "delay_requirement",
    "max_epsilon_range": "max_epsilon_range",
    "processing_capacity_range": "processing_capacity_range",
    "link_capacity_range": "link_capacity_range",
    "deltaF": "deltaF"
}
print()
# Load the configuration
config = load_config('config.json')

# Validate configuration values
def validate_config(config: dict):
    required_keys = CONFIG_KEYS.values()
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")

    # Add specific validation for ranges if needed
    for key, value in config.items():
        if isinstance(value, list) and len(value) == 2 and value[0] > value[1]:
            raise ValueError(f"Invalid range for {key}: {value}")

validate_config(config)

def generate_users(num_users: int) -> Dict[int, User]:
    users = {}
    for i in range(num_users):
        x_y_location = np.random.uniform(config[CONFIG_KEYS["x_y_location_range"]][0], config[CONFIG_KEYS["x_y_location_range"]][1], 2)
        z_location = 0
        real_location = np.concatenate((x_y_location, [z_location]))
        rate_requirement = np.random.uniform(config[CONFIG_KEYS["rate_requirement"]][0], config[CONFIG_KEYS["rate_requirement"]][1])
        delay_requirement = np.random.uniform(config[CONFIG_KEYS["delay_requirement"]][0], config[CONFIG_KEYS["delay_requirement"]][1])
        max_epsilon = np.random.uniform(config[CONFIG_KEYS["max_epsilon_range"]][0], config[CONFIG_KEYS["max_epsilon_range"]][1])  # Updated key
        user = User(i, rate_requirement, delay_requirement, max_epsilon, real_location)
        users[i] = user
    return users

def generate_nodes(num_nodes: int) -> Dict[int, Node]:
    nodes = {}
    for i in range(num_nodes):
        x_y_location = np.random.uniform(config[CONFIG_KEYS["x_y_location_range"]][0], config[CONFIG_KEYS["x_y_location_range"]][1], 2)
        z_location = np.random.uniform(config[CONFIG_KEYS["z_location_range"]][0], config[CONFIG_KEYS["z_location_range"]][1], 1)
        location = np.concatenate((x_y_location, z_location))
        node_type = np.random.choice(['G', 'D'], p=[1/3, 2/3])  # Randomly assign 'G' (Ground) or 'D' (Drone) such that 1 third are 'G' and 2 thirds are 'D'
        if i==0:
            node_type = 'G'
        processing_capacity = np.random.uniform(config[CONFIG_KEYS["processing_capacity_range"]][0], config[CONFIG_KEYS["processing_capacity_range"]][1]) if node_type == 'G' else 0.0
        node = Node(i, location, processing_capacity, node_type)
        nodes[i] = node
    return nodes

def calculate_distance_penalty(distance: float, deltaF: float) -> float:
    """Calculate the distance penalty for link capacity.

    Args:
        distance (float): Distance between two nodes.
        deltaF (float): Threshold distance for capacity adjustment.

    Returns:
        float: Distance penalty (minimum 0.5).
    """
    if distance >= np.sqrt(2) * deltaF:
        return 0.5
    return max(1 - (distance / (np.sqrt(2) * deltaF)), 0.5)


def create_link(node_1_id: int, node_2_id: int, nodes: Dict[int, Node], links: Dict[int, Link], existing_links: set) -> Link:
    """Create a link between two nodes with calculated capacity.

    Args:
        node_1_id (int): ID of the first node.
        node_2_id (int): ID of the second node.
        nodes (Dict[int, Node]): Dictionary of nodes.
        links (Dict[int, Link]): Dictionary of existing links.
        existing_links (set): Set of existing link pairs.

    Returns:
        Link: The created link object.
    """
    node_1 = nodes[node_1_id]
    node_2 = nodes[node_2_id]
    distance = np.linalg.norm(node_1.real_location - node_2.real_location)
    deltaF = config[CONFIG_KEYS["deltaF"]]
    distance_penalty = calculate_distance_penalty(distance, deltaF)
    link_capacity = np.random.uniform(
        config[CONFIG_KEYS["link_capacity_range"]][0],
        config[CONFIG_KEYS["link_capacity_range"]][1]
    ) * distance_penalty
    link = Link(len(links), node_1, node_2, link_capacity)
    existing_links.add((node_1_id, node_2_id))
    return link


def generate_links(num_links: int, nodes: Dict[int, Node]) -> Dict[int, Link]:
    """Generate links between nodes without visualization.

    Args:
        num_links (int): Number of links to generate.
        nodes (Dict[int, Node]): Dictionary of nodes.

    Returns:
        Dict[int, Link]: Dictionary of generated links.
    """
    from scipy.spatial import distance

    links = {}
    existing_links = set()
    node_ids = list(nodes.keys())
    node_degrees = {node_id: 0 for node_id in node_ids}  # Track the degree of each node

    # Precompute distances between all pairs of nodes
    distances = [
        (distance.euclidean(nodes[i].real_location, nodes[j].real_location), i, j)
        for i in node_ids for j in node_ids if i < j
    ]
    distances.sort()  # Sort by distance (ascending)

    # Step 2: Ensure all nodes are connected (one path between any two nodes)
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id):
        if parent[node_id] != node_id:
            parent[node_id] = find(parent[node_id])  # Path compression
        return parent[node_id]

    def union(node_1_id, node_2_id):
        root1 = find(node_1_id)
        root2 = find(node_2_id)
        if root1 != root2:
            parent[root2] = root1

    # Step 2: Ensure all nodes are connected
    for dist, node_1_id, node_2_id in distances:
        if find(node_1_id) != find(node_2_id):
            link = create_link(node_1_id, node_2_id, nodes, links, existing_links)
            links[link.id] = link
            union(node_1_id, node_2_id)

    # Step 1: Add additional links until num_links is used
    for dist, node_1_id, node_2_id in distances:
        if len(links) >= num_links:
            break
        if (node_1_id, node_2_id) not in existing_links and (node_2_id, node_1_id) not in existing_links:
            # Check if either node exceeds the degree limit
            if node_degrees[node_1_id] < 3 and node_degrees[node_2_id] < 3:
                link = create_link(node_1_id, node_2_id, nodes, links, existing_links)
                links[link.id] = link
                node_degrees[node_1_id] += 1
                node_degrees[node_2_id] += 1

    return links

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from data_structures import Node, Link
from matplotlib.cm import get_cmap
def plot_graph(nodes: Dict[int, Node], links: Dict[int, Link], title: str):
    """Plot the graph with nodes and links, assigning a different color to each link.

    Args:
        nodes (Dict[int, Node]): Dictionary of nodes.
        links (Dict[int, Link]): Dictionary of links.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    cmap = get_cmap("tab10")  # Use a colormap (e.g., tab10, viridis, etc.)
    colors = [cmap(i % 10) for i in range(len(links))]  # Generate colors for links

    # Plot nodes
    for node in nodes.values():
        x, y = node.real_location[:2]
        plt.scatter(x, y, label=f"Node {node.id}", s=100)
        plt.text(x, y, f"{node.id}", fontsize=9, ha='right')

    # Plot links with different colors
    for idx, link in enumerate(links.values()):
        x1, y1 = link.node_1.real_location[:2]
        x2, y2 = link.node_2.real_location[:2]
        plt.plot([x1, x2], [y1, y2], linestyle='--', color=colors[idx], alpha=0.7)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # plt.legend(loc="upper right", fontsize=8)  # Restoring the legend
    plt.grid(True)
    plt.show()
    
def generate_links_with_visualization(num_links: int, nodes: Dict[int, Node]) -> Dict[int, Link]:
    """Generate links between nodes and visualize the process interactively.

    Args:
        num_links (int): Number of links to generate.
        nodes (Dict[int, Node]): Dictionary of nodes.

    Returns:
        Dict[int, Link]: Dictionary of generated links.
    """
    from scipy.spatial import distance

    links = {}
    existing_links = set()
    node_ids = list(nodes.keys())
    node_degrees = {node_id: 0 for node_id in node_ids}  # Track the degree of each node

    # Precompute distances between all pairs of nodes
    distances = [
        (distance.euclidean(nodes[i].real_location, nodes[j].real_location), i, j)
        for i in node_ids for j in node_ids if i < j
    ]
    distances.sort()  # Sort by distance (ascending)

    # Step 2: Ensure all nodes are connected (one path between any two nodes)
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id):
        if parent[node_id] != node_id:
            parent[node_id] = find(parent[node_id])  # Path compression
        return parent[node_id]

    def union(node_1_id, node_2_id):
        root1 = find(node_1_id)
        root2 = find(node_2_id)
        if root1 != root2:
            parent[root2] = root1

    print("Step 2: Ensuring all nodes are connected")
    for dist, node_1_id, node_2_id in distances:
        if find(node_1_id) != find(node_2_id):
            link = create_link(node_1_id, node_2_id, nodes, links, existing_links)
            links[link.id] = link
            union(node_1_id, node_2_id)
            plot_graph(nodes, links, f"Step 2: Connecting Disconnected Components (Links: {len(links)})")

    # Step 1: Connect the closest nodes until num_links is used
    print("Step 1: Adding additional links if needed")
    for dist, node_1_id, node_2_id in distances:
        if len(links) >= num_links:
            break
        if (node_1_id, node_2_id) not in existing_links and (node_2_id, node_1_id) not in existing_links:
            # Check if either node exceeds the degree limit
            if node_degrees[node_1_id] < 3 and node_degrees[node_2_id] < 3:
                link = create_link(node_1_id, node_2_id, nodes, links, existing_links)
                links[link.id] = link
                node_degrees[node_1_id] += 1
                node_degrees[node_2_id] += 1
                plot_graph(nodes, links, f"Step 1: Connecting Closest Nodes (Links: {len(links)})")

    # Final plot of the graph
    print("Final Graph Visualization")
    plot_graph(nodes, links, "Final Graph Visualization")
    return links

#test if running the script as main
if __name__ == "__main__":
    import numpy as np
    from data_structures import Node, Link
    from generate_data import generate_nodes, generate_links_with_visualization

    # Configuration for testing
    NUM_NODES = 15  # Number of nodes to generate
    NUM_LINKS = 0  # Number of links to generate

    # Generate nodes
    nodes = generate_nodes(NUM_NODES)

    # Generate links with visualization
    links = generate_links_with_visualization(NUM_LINKS, nodes)
# %%
