import json
import matplotlib.pyplot as plt
import numpy as np
from utilities.analyze_results_tools import analyze_results_multiple
import os
from datetime import datetime   
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import webbrowser
from threading import Timer
from typing import List, Dict
from matplotlib import cm  # Add this import for colormap
from matplotlib.colors import ListedColormap

custom_labels = {
        "Privacy Maximization - Satisfied Latency": "PM-SL",
        "Privacy Satisfaction - Satisfied Latency": "PS-SL",
        "Privacy Satisfaction - Best Effort Latency": "PS-BEL",
        "Privacy Cancellation - Satisfied Latency": "PC-SL",
        "Privacy Cancellation - Best Effort Latency": "PC-BEL",
    }

# Add the new metrics to the metrics_to_plot dictionary
metrics_to_plot = {
    "average_delay_per_user": "Delay (s)",
    "average_throughput_per_user": "Throughput (Mbps)",
    "Total_throughput": "Total Throughput (Mbps)",
    "percentage_served_users": "Served Users (%)",
    "percentage_average_RBs_load": "RBs Load (%)",
    "percentage_average_processing_utilization": "Processing Utilization (%)",
    "average_epsilon_variation": "Epsilon Variation",
    "percentage_average_link_utilization": "Link Utilization (%)",
    "average_link_utilization": "Average Link Utilization",
    "percentage_link_usage": "Link Usage (%)",
    "served_users": "Served Users",
    "average_delay_violation": "Average Delay Violation (s)",
    # New execution time metrics
    "execution_time_total": "Total Execution Time (s)",
    "execution_time_step0": "Step 0 Execution Time (s)",
    "execution_time_step1": "Step 1 Execution Time (s)"
}

# Generate a color map for all scenarios
def get_color_map(scenarios, format="matplotlib"):
    """
    Generate a color map for all scenarios.

    Args:
        scenarios (List[Dict[str, str]]): List of scenarios.
        format (str): The format of the color map. Options are "matplotlib" or "plotly".

    Returns:
        Dict[str, Union[tuple, str]]: A dictionary mapping scenario names to colors.
    """
    # Define a custom list of colors (blue, red, yellow, etc.)
    custom_colors = ['blue', 'red', 'orange', 'green', 'purple', 'yellow']
    # Create a colormap from the custom colors
    cmap = ListedColormap(custom_colors[:len(scenarios)])

    if format == "matplotlib":
        # Return colors as (r, g, b) tuples for Matplotlib
        return {
            scenario["name"]: cmap(i / (len(scenarios) - 1))[:3]  # Extract only the RGB values
            for i, scenario in enumerate(scenarios)
        }
    elif format == "plotly":
        # Return colors as 'rgb(r,g,b)' strings for Plotly/Dash
        return {
            scenario["name"]: f"rgb({int(cmap(i / (len(scenarios) - 1))[0] * 255)},"
                              f"{int(cmap(i / (len(scenarios) - 1))[1] * 255)},"
                              f"{int(cmap(i / (len(scenarios) - 1))[2] * 255)})"
            for i, scenario in enumerate(scenarios)
        }
    else:
        raise ValueError("Invalid format. Use 'matplotlib' or 'plotly'.")
    
def plot_results(aggregated_results, scenarios: List[Dict[str, str]], metrics_to_plot: Dict[str, str], custom_labels: Dict[str, str] = custom_labels):
    """
    Plot the aggregated results for each metric dynamically based on scenarios.

    Args:
        aggregated_results (dict): Aggregated results containing metrics for each number of users.
        scenarios (List[Dict[str, str]]): List of scenarios to plot.
        metrics_to_plot (Dict[str, str]): Dictionary mapping metrics to their labels.
        custom_labels (Dict[str, str], optional): Dictionary mapping scenario names to custom legend labels.
    """
    color_map = get_color_map(scenarios, format="matplotlib")  # Generate the color map
    # Define a list of distinct markers
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'X', '+'] 

    for metric, y_label in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        num_users = sorted(aggregated_results.keys())

        # Prepare data for each scenario dynamically
        scenario_means = {scenario["name"]: [] for scenario in scenarios}
        scenario_cis = {scenario["name"]: [] for scenario in scenarios}
        for n_users in num_users:
            for scenario in scenarios:
                scenario_name = scenario["name"]
                scenario_data = aggregated_results[n_users].get(scenario_name)

                if scenario_data and metric in scenario_data:
                    mean = scenario_data[metric]["mean"]
                    ci = scenario_data[metric]["confidence_interval"]
                    if mean is not None and not np.isnan(mean):
                        scenario_means[scenario_name].append(mean)
                        scenario_cis[scenario_name].append(ci)
                    else:
                        scenario_means[scenario_name].append(0)
                        scenario_cis[scenario_name].append((0, 0))
                else:
                    scenario_means[scenario_name].append(0)
                    scenario_cis[scenario_name].append((0, 0))

        # Plot each scenario dynamically
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario["name"]
            means = scenario_means[scenario_name]

            # Skip scenarios with constant 0 values
            if all(value == 0 for value in means):
                continue

            label = custom_labels.get(scenario_name, scenario_name) if custom_labels else scenario_name
            # Select marker based on scenario index, cycling through the list
            marker = markers[i % len(markers)] 
            plot_error_bars(
                num_users,
                means,
                scenario_cis[scenario_name],
                label=label,
                fmt=f"-{marker}", # Combine line style '-' with the selected marker
                color=color_map[scenario_name],  # Use the unified color map
                linewidth=3 # Keep increased line width
            )

        # Add labels, legend, and grid
        plt.xlabel('Number of Users', fontsize=18, fontweight='bold') # Increase font size and make bold
        plt.ylabel(y_label, fontsize=18, fontweight='bold') # Increase font size and make bold
        # plt.title(f'{y_label} as a function of the Number of Users') # Temporarily remove title

        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            # Set font properties for legend labels to bold
            legend_prop = {'weight': 'bold', 'size': 20}
            plt.legend(handles=handles, labels=labels, prop=legend_prop, title_fontproperties={'weight':'bold'}) # Make legend labels bold

        plt.grid(True)
        # Ensure x-axis ticks are integers based on the provided user counts
        plt.xticks(num_users)

        # Increase the thickness of the ticks and make labels bold on both axes
        plt.tick_params(axis='both', which='major', width=2, labelsize=20) # Adjust width and labelsize as needed
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight('bold')


        # Save the plot in both EPS and PNG formats
        folder_name = datetime.now().strftime("results_%Y-%m-%d_%Hh%M")
        save_dir = os.path.join(os.path.dirname(__file__), folder_name)
        os.makedirs(save_dir, exist_ok=True)
        # Sanitize metric name for use in filename
        safe_metric_name = metric.replace(' ', '_').replace('/', '_')

        # Define base path without extension
        base_save_path = os.path.join(save_dir, f"{safe_metric_name}_plot")

        # Save as EPS (Encapsulated PostScript) - often preferred for publications
        eps_save_path = f"{base_save_path}.eps"
        plt.savefig(eps_save_path, format='eps', bbox_inches='tight')
        print(f"Plot saved to {eps_save_path}")

        # Save as PNG (or another format like PDF if preferred)
        png_save_path = f"{base_save_path}.png"
        plt.savefig(png_save_path, format='png', bbox_inches='tight', dpi=300) # Added dpi for better PNG resolution
        print(f"Plot saved to {png_save_path}")
    #save the current config file in the results directory by copying it from the current directory using system instead of loading into python
    
    import shutil    
    # Assuming `folder_name` and `aggregated_results` are defined earlier in your code
    shutil.copy('config.json', folder_name)  # Replacing 'copy' with 'shutil.copy'
      # Save the aggregated results to a JSON file
    # Convert numpy values to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.number):
            return obj.item()  # Convert numpy types like float32 to Python native types
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    # Apply conversion to aggregated_results
    serializable_results = convert_for_json(aggregated_results)
    
    with open(os.path.join(folder_name, 'aggregated_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    return save_dir
    
def plot_error_bars(num_users, means, confidence_intervals, label, fmt, color, linewidth=1.5): # Add linewidth parameter
    """
    Plot error bars for the given data.

    Args:
        num_users (list): List of user counts.
        means (list): List of mean values.
        confidence_intervals (list): List of confidence intervals (tuples of lower and upper bounds).
        label (str): Label for the plot.
        fmt (str): Format string for the plot markers and line style.
        color (str): Color for the plot line and markers.
        linewidth (float): Width of the plot line.
    """
    if any(m is not None for m in means):
        means_array = np.array(means)
        ci_array = np.array(confidence_intervals)

        # Calculate error bar lengths directly from confidence intervals
        # Note: This assumes confidence_intervals contains valid (lower, upper) tuples
        # and does not explicitly handle NaNs or ensure bounds are relative to the mean.
        err_lower = means_array - ci_array[:, 0]
        err_upper = ci_array[:, 1] - means_array
        
        # Combine lower and upper errors for plt.errorbar
        yerr = np.array([err_lower, err_upper])

        plt.errorbar(
            num_users,
            means_array, 
            yerr=yerr,
            fmt=fmt, # fmt controls marker and line style
            label=label,
            capsize=5,
            color=color,
            linewidth=linewidth # Set the line width
        )

def plot_user_and_node_positions(users, nodes):
    """
    Plot the x and y positions of users and nodes.

    Args:
        users (Dict[int, User]): Dictionary of User objects.
        nodes (Dict[int, Node]): Dictionary of Node objects.
    """
    plt.figure(figsize=(8, 6))

    # Extract user positions
    user_positions = [user.real_location for user in users.values()]
    user_positions = np.array(user_positions)

    # Extract node positions
    node_positions = [node.real_location for node in nodes.values()]
    node_positions = np.array(node_positions)

    # Plot user positions
    plt.scatter(user_positions[:, 0], user_positions[:, 1], c='blue', label='Users', marker='o')

    # Plot node positions
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c='red', label='Nodes', marker='^')

    # Add labels, legend, and grid
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Positions of Users and Nodes')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show(block=False)
    plt.pause(0.1)  # Allow the plot to render

def plot_plotly(aggregated_results, scenarios: List[Dict[str, str]], metrics_to_plot: Dict[str, str]):
    """
    Create interactive plots using Plotly for the aggregated results, all on one page.

    Args:
        aggregated_results (dict): Aggregated results containing metrics for each number of users.
        scenarios (List[Dict[str, str]]): List of scenarios to plot.
        metrics_to_plot (Dict[str, str]): Dictionary mapping metrics to their labels.
    """
    # Create a subplot grid with one row per metric
    rows = len(metrics_to_plot)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{y_label} vs Number of Users" for y_label in metrics_to_plot.values()]
    )

    num_users = sorted(aggregated_results.keys())

    for i, (metric, y_label) in enumerate(metrics_to_plot.items(), start=1):
        for scenario in scenarios:
            scenario_name = scenario["name"]
            means = []

            for n_users in num_users:
                scenario_data = aggregated_results[n_users].get(scenario_name)
                if scenario_data and metric in scenario_data:
                    mean = scenario_data[metric]["mean"]
                    means.append(mean if mean is not None else 0)
                else:
                    means.append(0)

            # Skip scenarios with constant 0 values
            if all(value == 0 for value in means):
                continue

            # Add a trace for the scenario
            fig.add_trace(
                go.Scatter(
                    x=num_users,
                    y=means,
                    mode='lines+markers',
                    name=scenario_name,
                    legendgroup=scenario_name
                ),
                row=i,
                col=1
            )

        # Update y-axis title for the subplot
        fig.update_yaxes(title_text=y_label, row=i, col=1)

    # Update layout
    fig.update_layout(
        height=300 * rows,  # Adjust height based on the number of rows
        title_text="Aggregated Results",
        showlegend=True,
        template="plotly_white"
    )

    # Show the interactive plot
    fig.show()

def plot_dash(aggregated_results, scenarios: List[Dict[str, str]], metrics_to_plot: Dict[str, str]):
    """
    Create a Dash app for interactive visualization of aggregated results.

    Args:
        aggregated_results (dict): Aggregated results containing metrics for each number of users.
        scenarios (List[Dict[str, str]]): List of scenarios to plot.
        metrics_to_plot (Dict[str, str]): Dictionary mapping metrics to their labels.
    """
    # Initialize Dash app
    app = dash.Dash(__name__)

    # Create a dropdown for metric selection
    dropdown_metric = dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': y_label, 'value': metric} for metric, y_label in metrics_to_plot.items()],
        value=list(metrics_to_plot.keys())[0],  # Default to the first metric
        clearable=False
    )

    # Create a dropdown for scenario selection
    dropdown_scenario = dcc.Dropdown(
        id='scenario-dropdown',
        options=[{'label': scenario["name"], 'value': scenario["name"]} for scenario in scenarios] +
                [{'label': 'All Scenarios', 'value': 'all'}],
        value='all',  # Default to all scenarios
        clearable=False
    )

    # Layout of the app
    app.layout = html.Div([
        html.H1("Interactive Simulation Results"),
        html.Div([
            html.Label("Select Metric:"),
            dropdown_metric,
        ]),
        html.Div([
            html.Label("Select Scenario:"),
            dropdown_scenario,
        ]),
        dcc.Graph(id='metric-plot')
    ])

    # Callback to update the plot based on the selected metric and scenario
    @app.callback(
        dash.dependencies.Output('metric-plot', 'figure'),
        [dash.dependencies.Input('metric-dropdown', 'value'),
         dash.dependencies.Input('scenario-dropdown', 'value')]
    )
    def update_plot(selected_metric, selected_scenario):
        num_users = sorted(aggregated_results.keys())
        fig = go.Figure()

        # Use the color map for consistent colors
        color_map = get_color_map(scenarios, format="plotly")

        # Filter scenarios based on selection
        filtered_scenarios = scenarios if selected_scenario == 'all' else [
            scenario for scenario in scenarios if scenario["name"] == selected_scenario
        ]

        for scenario in filtered_scenarios:
            scenario_name = scenario["name"]
            means = []
            for n_users in num_users:
                scenario_data = aggregated_results[n_users].get(scenario_name)
                if scenario_data is not None and selected_metric in scenario_data:
                    mean = scenario_data[selected_metric].get("mean", 0)
                    means.append(mean)
                else:
                    means.append(0)

            # Skip scenarios with constant 0 values
            if all(value == 0 for value in means):
                continue

            # Add a trace for the scenario
            fig.add_trace(go.Scatter(
                x=num_users,
                y=means,
                mode='lines+markers',
                name=scenario_name,
                line=dict(color=color_map[scenario_name])  # Use consistent colors
            ))

        # Update layout
        fig.update_layout(
            title=f"{metrics_to_plot[selected_metric]} vs Number of Users",
            xaxis_title="Number of Users",
            yaxis_title=metrics_to_plot[selected_metric],
            template="plotly_white"
        )
        return fig

    # Automatically open the browser and run the Dash app without blocking
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    Timer(1, open_browser).start()
    app.run(debug=False, use_reloader=False)


