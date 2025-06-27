import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
G.add_node("AI/ML-Driven Data and Model Lifecycle in 6G", style="central")

# Opportunities
G.add_node("Opportunities AI/ML Brings to 6G Ecosystem", style="opportunities")
G.add_edge("AI/ML-Driven Data and Model Lifecycle in 6G", "Opportunities AI/ML Brings to 6G Ecosystem")
G.add_node("Enhances efficiency and business opportunities for stakeholders", style="opportunities_detail")
G.add_node("Enables zero-touch management and autonomous operations", style="opportunities_detail")
G.add_node("Supports intent-based networking and real-time decision-making", style="opportunities_detail")
G.add_edges_from([
    ("Opportunities AI/ML Brings to 6G Ecosystem", "Enhances efficiency and business opportunities for stakeholders"),
    ("Opportunities AI/ML Brings to 6G Ecosystem", "Enables zero-touch management and autonomous operations"),
    ("Opportunities AI/ML Brings to 6G Ecosystem", "Supports intent-based networking and real-time decision-making")
])

# Challenges
G.add_node("Challenges in Data Availability for AI/ML in 6G", style="challenges")
G.add_edge("AI/ML-Driven Data and Model Lifecycle in 6G", "Challenges in Data Availability for AI/ML in 6G")
G.add_node("Need for high-quality, large-scale datasets", style="challenges_detail")
G.add_node("Scarcity of realistic datasets for RAN scenarios", style="challenges_detail")
G.add_node("Fragmentation and lack of end-to-end system behavior", style="challenges_detail")
G.add_edges_from([
    ("Challenges in Data Availability for AI/ML in 6G", "Need for high-quality, large-scale datasets"),
    ("Challenges in Data Availability for AI/ML in 6G", "Scarcity of realistic datasets for RAN scenarios"),
    ("Challenges in Data Availability for AI/ML in 6G", "Fragmentation and lack of end-to-end system behavior")
])

# Existing Initiatives
G.add_node("Existing Data Initiatives", style="initiatives")
G.add_edge("AI/ML-Driven Data and Model Lifecycle in 6G", "Existing Data Initiatives")
G.add_node("International Data Spaces (IDS) and Gaia-X", style="initiatives_detail")
G.add_node("Need for a Gaia-X-like framework for 6G", style="initiatives_detail")
G.add_edges_from([
    ("Existing Data Initiatives", "International Data Spaces (IDS) and Gaia-X"),
    ("Existing Data Initiatives", "Need for a Gaia-X-like framework for 6G")
])

# Model Testing
G.add_node("Model Testing and Validation Issues", style="testing")
G.add_edge("AI/ML-Driven Data and Model Lifecycle in 6G", "Model Testing and Validation Issues")
G.add_node("Difficulty in validating models in realistic environments", style="testing_detail")
G.add_node("Tools like AI Gym lack complexity", style="testing_detail")
G.add_node("MLOps evolution for 6G testbeds, digital twins, and RL safety", style="testing_detail")
G.add_edges_from([
    ("Model Testing and Validation Issues", "Difficulty in validating models in realistic environments"),
    ("Model Testing and Validation Issues", "Tools like AI Gym lack complexity"),
    ("Model Testing and Validation Issues", "MLOps evolution for 6G testbeds, digital twins, and RL safety")
])

# Trustworthy AI
G.add_node("Need for Trustworthy AI in 6G", style="trustworthy")
G.add_edge("AI/ML-Driven Data and Model Lifecycle in 6G", "Need for Trustworthy AI in 6G")
G.add_node("Ensuring no functionality breakage", style="trustworthy_detail")
G.add_node("Ensuring no security/privacy violations", style="trustworthy_detail")
G.add_node("Ensuring no unintended actions", style="trustworthy_detail")
G.add_edges_from([
    ("Need for Trustworthy AI in 6G", "Ensuring no functionality breakage"),
    ("Need for Trustworthy AI in 6G", "Ensuring no security/privacy violations"),
    ("Need for Trustworthy AI in 6G", "Ensuring no unintended actions")
])

# Draw the graph with enhanced styling
pos = nx.circular_layout(G)
plt.figure(figsize=(14, 10))

# Draw nodes with color coding
node_colors = []
node_sizes = []
for node in G.nodes:
    if "opportunities" in node:
        node_colors.append("#4CAF50")  # Green for opportunities
        node_sizes.append(4000)
    elif "challenges" in node:
        node_colors.append("#FF5722")  # Orange for challenges
        node_sizes.append(4000)
    elif "initiatives" in node:
        node_colors.append("#2196F3")  # Blue for initiatives
        node_sizes.append(4000)
    elif "testing" in node:
        node_colors.append("#9C27B0")  # Purple for testing
        node_sizes.append(4000)
    elif "trustworthy" in node:
        node_colors.append("#FFC107")  # Yellow for trustworthy AI
        node_sizes.append(4000)
    else:
        node_colors.append("#607D8B")  # Default gray
        node_sizes.append(3000)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors="black")
nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20, edge_color="gray", connectionstyle="arc3,rad=0.2")

# Adjust labels to fit within node boundaries
labels = {}
for node in G.nodes:
    words = node.split(" ")
    wrapped_label = "\n".join([" ".join(words[i:i+3]) for i in range(0, len(words), 3)])
    labels[node] = wrapped_label

nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black", font_weight="bold")

# Add title and background
plt.title("AI/ML-Driven Data and Model Lifecycle in 6G", fontsize=18, fontweight="bold", color="#3E2723")
plt.gca().set_facecolor("#F5F5F5")
plt.axis("off")
plt.show()
