import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objs as go
import numpy as np
import json
import community as community_louvain  # 导入 Louvain 社区发现算法

# 创建 Dash 应用
app = dash.Dash(__name__)

# 文件路径
network_file = 'edges.txt'
dates_file = 'paper_dates.txt'
pagerank_file = 'HepTh_pagerank_results.json'

def read_network(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            source, target = line.strip().split()
            G.add_edge(source, target)
    return G

def read_dates(file_path):
    dates = {}
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            parts = line.strip().split()
            paper_id = parts[0]
            date = parts[1]
            dates[str(idx + 1)] = (paper_id, date)
    return dates

def read_paper_details(file_path):
    details = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            paper_id = parts[0]
            title = parts[1]
            authors = parts[2]
            details[paper_id] = (title, authors)
    return details

def create_graph(important_nodes, G):
    return G.subgraph(important_nodes).copy()

def calculate_fixed_positions(G, layout_type):
    if layout_type == 'spring':
        pos = nx.spring_layout(G, dim=3, seed=42)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, dim=3)
    else:  # fruchterman_reingold
        pos = nx.fruchterman_reingold_layout(G, dim=3)
    fixed_positions = {node: (pos[node][0], pos[node][1], pos[node][2]) for node in G.nodes()}
    return fixed_positions

# 全局固定位置字典和布局类型
G_full = read_network(network_file)
spring_fixed_positions = calculate_fixed_positions(G_full, 'spring')

def generate_figure(subgraph, color_mode, dates, layout_type):
    if layout_type == 'spring':
        pos = {node: spring_fixed_positions[node] for node in subgraph.nodes()}
    else:
        pos = calculate_fixed_positions(subgraph, layout_type)

    node_x = np.array([pos[node][0] for node in subgraph.nodes()])
    node_y = np.array([pos[node][1] for node in subgraph.nodes()])
    node_z = np.array([pos[node][2] for node in subgraph.nodes()])

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in subgraph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    if color_mode == 'year':
        years = np.array([int(dates[node][1].split('-')[0]) for node in subgraph.nodes()])
        norm_years = (years - years.min()) / (years.max() - years.min())
        colors = [f'rgba({int(173 + (0 - 173) * n)}, {int(216 + (0 - 216) * n)}, {int(230 + (255 - 230) * n)}, 1)' for n in norm_years]
    elif color_mode == 'louvain':
        partition = community_louvain.best_partition(subgraph.to_undirected())
        communities = np.array([partition[node] for node in subgraph.nodes()])
        unique_communities = np.unique(communities)
        colors = [f'rgba({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)}, 1)' for _ in unique_communities]
        community_color_map = {com: color for com, color in zip(unique_communities, colors)}
        colors = [community_color_map[partition[node]] for node in subgraph.nodes()]
    else:
        colors = ['rgba(0, 0, 255, 1)'] * len(subgraph.nodes())

    node_degrees = np.array([subgraph.degree[node] for node in subgraph.nodes()])
    node_sizes = 5 + 45 * (node_degrees - node_degrees.min()) / (node_degrees.max() - node_degrees.min())

    figure = go.Figure()

    figure.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=1, color='rgba(128, 128, 128, 0.8)'),
        hoverinfo='none',
        showlegend=False
    ))

    figure.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=node_sizes, color=colors),
        text=list(subgraph.nodes()),
        hoverinfo='text',
        showlegend=False
    ))

    if color_mode == 'year':
        figure.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                size=4,
                color=years,
                colorscale='Blues',
                colorbar=dict(
                    title="Year",
                    tickvals=np.linspace(years.min(), years.max(), 5),
                    ticktext=[str(int(year)) for year in np.linspace(years.min(), years.max(), 5)]
                )
            ),
            hoverinfo='none',
            showlegend=False
        ))

    figure.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    ),
    margin=dict(l=0, r=0, b=0, t=0))

    return figure

def calculate_network_stats(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    return num_nodes, num_edges, avg_degree

yearRange = [1993, 1995, 1996, 1998, 1999, 2000, 1992, 1994, 1997, 2001, 2002]

# Dash 应用布局
app.layout = html.Div([
    html.Div([
        html.H3("Control Panel"),
        html.P('Data size:'),
        dcc.Slider(
            id='data-slider',
            min=1,
            max=10,
            marks={i: f'{i*10}%' for i in range(1, 11)},
            value=1  # 默认值为 10%
        ),
        html.P('Layout algorithm:'),
        dcc.RadioItems(
            id='layout-selector',
            options=[
                {'label': 'Spring Layout', 'value': 'spring'},
                {'label': 'Fruchterman Reingold Layout', 'value': 'fruchterman_reingold'},
                {'label': 'Kamada Kawai Layout', 'value': 'kamada_kawai'}
            ],
            value='spring',  # 默认布局
            labelStyle={'display': 'inline-block'}
        ),
        html.P('Years showing in the graph:'),
        dcc.RangeSlider(
            id='year-slider',
            min=min(yearRange),
            max=max(yearRange),
            step=1,
            marks={str(year): str(year) for year in range(min(yearRange), max(yearRange)+1)},
            value=[min(yearRange), max(yearRange)],
        ),
        html.P('Node Degree Threshold:'),
        dcc.Slider(
            id='degree-slider',
            min=0,
            max=8,
            marks={i: f'{i}' for i in range(9)},
            value=0  # 默认值为 0
        ),
        html.Div(id='degree-value', style={'margin-top': '10px'}),
        html.P('Color mode:'),
        dcc.RadioItems(
            id='color-mode-selector',
            options=[
                {'label': 'By Year', 'value': 'year'},
                {'label': 'By Louvain Community', 'value': 'louvain'},
                {'label': 'Single Color', 'value': 'single'}
            ],
            value='year',  # 默认颜色模式
            labelStyle={'display': 'inline-block'}
        ),
        html.H3("Network Statistics"),
        html.Div(id='network-stats'),
        html.H3("Node Information"),
        html.Div(id='node-info')
    ], style={'width': '30%', 'display': 'inline-block', 'height': '100vh', 'overflowY': 'scroll'}),
    
    dcc.Graph(id='network-graph', style={'width': '70%', 'display': 'inline-block', 'height': '100vh'})
])

@app.callback(
    Output('network-graph', 'figure'),
    Output('network-stats', 'children'),
    [Input('data-slider', 'value'),
     Input('layout-selector', 'value'),
     Input('year-slider', 'value'),
     Input('degree-slider', 'value'),
     Input('color-mode-selector', 'value')]
)
def update_graph(selected_percentage, layout_type, year_range, degree_threshold, color_mode):
    proportion = selected_percentage / 10  # 将滑块值转换为比例
    
    with open(pagerank_file, 'r') as file:
        data = json.load(file)
        pagerank = data['pagerank']
        important_nodes = data['important_nodes'][:int(len(data['important_nodes']) * proportion)]
    
    subgraph = create_graph(important_nodes, G_full)
    
    # Filter nodes based on selected year range
    dates = read_dates(dates_file)
    filtered_nodes = [node for node in subgraph.nodes() if int(dates[node][1].split('-')[0]) in range(year_range[0], year_range[1] + 1)]
    subgraph_filtered = subgraph.subgraph(filtered_nodes)
    
    # Filter nodes based on degree threshold
    final_nodes = [node for node in subgraph_filtered.nodes() if subgraph_filtered.degree[node] >= degree_threshold]
    subgraph_final = subgraph_filtered.subgraph(final_nodes)
    
    # Calculate network statistics
    num_nodes, num_edges, avg_degree = calculate_network_stats(subgraph_final)
    stats_info = [
        html.P(f'Number of nodes: {num_nodes}'),
        html.P(f'Number of edges: {num_edges}'),
        html.P(f'Average degree: {avg_degree:.2f}')
    ]
    
    return generate_figure(subgraph_final, color_mode, dates, layout_type), stats_info

@app.callback(
    Output('degree-value', 'children'),
    [Input('degree-slider', 'value')]
)
def update_degree_value(value):
    return f'  Hide nodes which degree < {value}'

@app.callback(
    Output('node-info', 'children'),
    [Input('network-graph', 'clickData')]
)
def display_node_info(clickData):
    if clickData and 'points' in clickData:
        node_text = clickData['points'][0].get('text', '')
        if node_text:
            dates = read_dates(dates_file)
            G = read_network(network_file)
            paper_details = read_paper_details('paper_details.txt')
            
            if node_text in dates:
                paper_id, date = dates[node_text]
                title, authors = paper_details.get(paper_id, ("Unknown Title", "Unknown Authors"))
                degree = G.degree[node_text]
                connected_papers = [dates[neighbor][0] for neighbor in G.neighbors(node_text) if neighbor in dates]
                return html.Div([
                    html.P(f"Node: {node_text}"),
                    html.P(f"Paper ID: {paper_id}"),
                    html.P(f"Date: {date}"),
                    html.P(f"Title: {title}"),
                    html.P(f"Authors: {authors}"),
                    html.P(f"Degree: {degree}"),
                    html.P(f"Connected Papers: {', '.join(connected_papers)}")
                ])
    return html.Div([
        html.P("Please click the node to show the node Information")
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
