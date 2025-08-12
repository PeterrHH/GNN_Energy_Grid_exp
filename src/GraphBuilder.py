import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import toml
import torch
from torch_geometric.utils import from_networkx
import json
import os
from torch_geometric.data import HeteroData

from utils import *



# TRANSMISSION_LINE_PATH = "case_studies/stylized_EU/inputs/transmission_lines.csv"
TRANSMISSION_LINE_PATH = "small_instance/inputs/transmission_lines.csv"
SAVE_PATH = "SimpleGridGraph.pt"
JSON_SAVE_PATH = "index_to_name.json"
TECHNOLOGIES = ['Coal', 'Gas', 'Lignite', 'Nuclear', 'Oil', 'SunPV', 'WindOff', 'WindOn']
TRADITIONAL_ENERGY = ['Coal', 'Gas', 'Lignite', 'Nuclear', 'Oil']
MAX_VALUE = 10000
DEFAULT_FEATURES = {
    'investment_cost': MAX_VALUE,
    'variable_cost': MAX_VALUE,
    'unit_capacity': 0,
    'ramping_rate': 0.0,
    'availability': 0
}



def read_input_data(base_path):

    demand_path = os.path.join(base_path,"inputs", "demand.csv")
    generation_availability_path = os.path.join(base_path,"inputs", "generation_availability.csv")  
    generation_path = os.path.join(base_path,"inputs", "generation.csv")
    scalars_path = os.path.join(base_path,"inputs", "scalars.toml")
    transmission_line_path  = os.path.join(base_path, "inputs", "transmission_lines.csv")
    demand_df    = pd.read_csv(demand_path)
    generation_availability_df = pd.read_csv(generation_availability_path)
    generation_df = pd.read_csv(generation_path)
    transmission_line_df = pd.read_csv(transmission_line_path)
    with open(scalars_path, 'r') as f:
        scalars = toml.load(f)

    value_of_lost_load = scalars['value_of_lost_load']
    relaxation = scalars['relaxation']

    return transmission_line_df, demand_df, generation_availability_df, generation_df, value_of_lost_load, relaxation


def build_graph(base_path, graph_save_path=SAVE_PATH, json_save_path=JSON_SAVE_PATH, directed=True, seperate_technology = True, plot = True, save = True):
    transmission_line_path  = os.path.join(base_path, "inputs", "transmission_lines.csv")
    production_path = os.path.join(base_path, "inputs", "generation.csv")
    transmission_df = pd.read_csv(transmission_line_path)
    production_df = pd.read_csv(production_path)

    G = nx.DiGraph()
    location_unique = production_df['location'].unique()

    for index, row in transmission_df.iterrows():
        print(f"Index: {index}")
        print(f"From: {row['from']}")
        print(f"To: {row['to']}")
        print(f"Export Capacity: {row['export_capacity']}")
        print(f"Import Capacity: {row['import_capacity']}")

        # Add vertices if not already added
        if not G.has_node(row['from']):
            G.add_node(row['from'])
            print(f"ADDING TO VERTEX: {row['from']}")

        if not G.has_node(row['to']):
            G.add_node(row['to'])
            print(f"ADDING TO VERTEX: {row['to']}")

        if seperate_technology:
            name = f"{row['from']}-{row['to']}"
            G.add_node(name)
            G.add_edge(name, row['from'], directed = True)
            G.add_edge(row['from'], name, directed = True)
            G.add_edge(name, row['to'], directed = True)
            G.add_edge(row['to'], name, directed = True)
        else:
            if directed:
                G.add_edge(row['from'], row['to'],directed = True)
                G.add_edge(row['to'], row['from'], directed = True)
            else:
                G.add_edge(row['from'], row['to'], directed = False)



    if seperate_technology:
        # For all existing  nodes, add a corresponding demand node and make a directed connection from the location node to demand node

        # for node in list(G.nodes):
        #     # Create a demand node name (e.g., "A" → "A_demand")
        #     demand_node = f"D_{node}"

        #     # Add the demand node
        #     G.add_node(demand_node)

        #     # Create a directed edge from the location node to the demand node
        #     G.add_edge(node, demand_node, directed=True)

        #     print(f"Added demand node {demand_node} and edge from {node} -> {demand_node}")
    
        for loc in location_unique:
            demand_node = f"D_{loc}"
            if not G.has_node(demand_node):
                G.add_node(demand_node)
                G.add_edge(loc, demand_node, directed = True)
                print(f"Added demand node {demand_node} and edge from {loc} -> {demand_node}")

        for index, row in production_df.iterrows():
            name = f"{row['technology']}_{row['location']}"
            if not G.has_node(name):
                G.add_node(name)
                G.add_edge(name,row["location"],directed = True)

    if directed:
        # data = from_networkx(G, group_edge_attrs=['capacity'])
        data = from_networkx(G)
    else:
        # data = from_networkx(G, group_edge_attrs=['export_capacity', 'import_capacity'])
        data = from_networkx(G)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    node_names = list(G.nodes())  
    data.name = node_names
    index_to_name = {i: name for i, name in enumerate(node_names)}
    print(f"Index to Name Mapping: {index_to_name}")
    if save:
        with open(json_save_path, "w") as f:
            json.dump(index_to_name, f)

        torch.save(data, graph_save_path)


    if plot:
        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(G)
        labels = {node: node for node in G.nodes()}  
        

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        plt.title("Transmission Grid Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return G, index_to_name


'''
Node Feature (N,D)

'''
def build_features(base_path,G, vertex_to_name):
    # Add features to the graph
    lines_df, demand_df, generation_availability_df, generation_df, value_of_lost_load, relaxation = read_input_data(base_path)

    pivoted = demand_df.pivot(index='time_step', columns='location', values='demand')
    demand_array = pivoted.values[:, :, np.newaxis]  # shape (T, N, D=1)

    print("Demand Shape (T, N, D):", demand_array.shape)

    location_order = [vertex_to_name[str(i)] for i in range(len(vertex_to_name))]

    generation_df['location'] = pd.Categorical(
        generation_df['location'],
        categories=location_order,
        ordered=True
    )
    generation_df = generation_df.sort_values('location').reset_index(drop=True)
    generation_df["availability"] = 1

    gen_features = generation_df.to_numpy() 

    T = demand_array.shape[0]
    N = demand_array.shape[1]
    gen_expanded = np.broadcast_to(gen_features, (T, N, 7))  # shape (T, 2, 7)
    # Step 3: Concatenate demand and generation along the feature axis
    final_array = np.concatenate([demand_array, gen_expanded], axis=-1)  # shape (T, 2, 8)

    selected_indices = [0, 4, 5, 7] # 4 is variable_cost 5 is unit capacity 7 is availability
    node_features = torch.tensor(final_array[:, :, selected_indices].astype(np.float32), dtype=torch.float32) # shape (T, 2, 6)

    # Build edge features
    edge_index = G.edge_index
    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    edge_to_index = {edge: i for i, edge in enumerate(edge_list)}  
    num_edges = len(edge_list)

    name_to_vertex = {v: k for k, v in vertex_to_name.items()}
    edge_features = torch.zeros((T, num_edges, 1), dtype=torch.float32) # Edge Feature size is 1.
    print(f"Edge Feature Shape: {edge_list}")
    for _, row in lines_df.iterrows():
        u, v = int(name_to_vertex[row['from']]), int(name_to_vertex[row['to']])
        import_capacity = row['import_capacity']
        export_capacity = row['export_capacity']
        if (u, v) in edge_to_index:
            edge_features[:, edge_to_index[(u, v)], 0] = export_capacity
        if (v, u) in edge_to_index:
            edge_features[:, edge_to_index[(v, u)], 0] = import_capacity


    return node_features, edge_features

def read_output_data(base_path):
    investment_path = os.path.join(base_path,"output", "investment.csv")
    line_flow_path = os.path.join(base_path,"output", "line_flow.csv")
    loss_of_load_path = os.path.join(base_path,"output", "loss_of_load.csv")
    production_path = os.path.join(base_path,"output", "production.csv")
    scalars_path = os.path.join(base_path,"output", "scalars.toml")
    with open(scalars_path, 'r') as f:
        scalars = toml.load(f)
        total_investment_cost = scalars['total_investment_cost']
        total_operational_cost = scalars['total_operational_cost']
        runtime = scalars['runtime']

    investment_df = pd.read_csv(investment_path)
    line_flow_df = pd.read_csv(line_flow_path)
    loss_of_load_df = pd.read_csv(loss_of_load_path)
    production_df = pd.read_csv(production_path)

    return investment_df, line_flow_df, loss_of_load_df, production_df, scalars

def build_ground_truth(base_path, G, vertex_to_name):
    '''
    At the moment, only build Ground Truth for: 
    1. Production per ndoe (In future + investment per node) When there are more technology it should be  Shape: (N,8)
            For now, it is [N,1] as we have 1 technology and we only consider production now
    2. Flow per edge Shape:(|E|,1)

    '''
    investment_df, line_flow_df, loss_of_load_df, production_df,output_scalars = read_output_data(base_path)

    # Get Node GT
    location_order = [vertex_to_name[str(i)] for i in range(len(vertex_to_name))]

    production_df['location'] = pd.Categorical(
        production_df['location'],
        categories=location_order,
        ordered=True
    )

    production_df = production_df.sort_values('location').reset_index(drop=True)
    pivoted = production_df.pivot(index='time_step', columns='location', values='production')
    
    pivoted = pivoted[location_order]  
    T, N = pivoted.shape
    node_gt = torch.tensor(pivoted.to_numpy().reshape(T, N, 1), dtype=torch.float32)

    # Get Edge GT
    edge_index = G.edge_index
    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    edge_to_index = {edge: i for i, edge in enumerate(edge_list)}  
    num_edges = len(edge_list)
    

    edge_gt = torch.zeros((T, num_edges, 1), dtype=torch.float32)
    name_to_vertex = {v: k for k, v in vertex_to_name.items()}

    for _, row in line_flow_df.iterrows():
        time_step = int(row['time_step'])
        u, v = int(name_to_vertex[row['from']]), int(name_to_vertex[row['to']])
        flow = row['flow']
        
        if (u, v) in edge_to_index:
            if flow >= 0:
                edge_gt[time_step-1, edge_to_index[(u, v)], 0] = flow
            else:
                edge_gt[time_step-1, edge_to_index[(u, v)], 0] = 0
                # assign negative flow to reverse edge if it exists
                if (v, u) in edge_to_index:
                    edge_gt[time_step-1, edge_to_index[(v, u)], 0] = -flow
        
    return node_gt, edge_gt.squeeze(2)


import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx

def plot_hetero_graph_from_data(node_feat, edge_feat,tech_to_index, loc_to_index, demand_to_index):
    index_to_tech = {v: k for k, v in tech_to_index.items()}
    index_to_loc = {v: k for k, v in loc_to_index.items()}
    index_to_demand = {v: k for k, v in demand_to_index.items()}
    print(f"Index to Tech {index_to_tech}")
    print(f"Index to Location {index_to_loc}")
    print(f"Index to Demand {index_to_demand}")
    G = nx.DiGraph()

    # Add nodes per type
    for node_type, feats in node_feat.items():
        print(f"Adding {node_type} nodes with shape {feats.shape}")
        if node_type == 'demand':
            for i in range(feats.size(1)):
                name = index_to_demand[i]
                G.add_node(f"{node_type}_{i}", node_type=node_type, label=name)
        elif node_type == 'technology':
            for i in range(feats.size(1)):
                name = index_to_tech[i]
                G.add_node(f"{node_type}_{i}", node_type=node_type, label = name)
        else:
            for i in range(feats.size(0)):
                if node_type == 'location':
                    name = index_to_loc[i]
                else:
                    name = f"{node_type}_{i}"
                G.add_node(f"{node_type}_{i}", node_type=node_type, label=name)

    # Helper to map indices to names
    def src_dst_names(src_idx, dst_idx, src_type, dst_type):
        return f"{src_type}_{src_idx}", f"{dst_type}_{dst_idx}"

    # Add edges for each relation
    for rel_name, edge_index in edge_feat.items():
        if rel_name == 'tech2loc':
            src_type, dst_type = 'technology', 'location'
        elif rel_name == 'loc2demand':
            src_type, dst_type = 'location', 'demand'
        elif rel_name == 'loc2flow':
            src_type, dst_type = 'location', 'flow'
        elif rel_name == 'flow2loc':
            src_type, dst_type = 'flow', 'location'
        else:
            continue  # skip unknown relations

        for src_idx, dst_idx in edge_index.t().tolist():
            src_name, dst_name = src_dst_names(src_idx, dst_idx, src_type, dst_type)
            
            G.add_edge(src_name, dst_name, relation=rel_name)
            G.add_edge(dst_name, src_name, relation=rel_name)

    # Color nodes by type
    color_map = {
        'technology': 'red',
        'location': 'blue',
        'demand': 'green',
        'flow': 'orange'
    }

    node_colors = [color_map[G.nodes[n]['node_type']] for n in G.nodes]

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # deterministic layout

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->')
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, font_size=8, labels = labels)

    plt.title("Heterogeneous Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def build_hetero_graph(base_path, use_investment_as_feature = False, plot = False):
    '''
    --- Node: 
    Technology  (#(Tech,Location), feat_size) now feat_size = 4 ['technology_name','cost', 'capacity', 'availability'] later will add investment, ramping related
    Location    (#Location,location_id?)
    Demand      (#Location, Demand)
    Flow_Node   (#Location_Pair*4, 1) # between 2 loc, there is A->Export_Cap->B, B->Import_Cap->B
    --- Edge:
    Technology   ->    Location
    Location     ->    Demand
    Location     ->    Location
    '''
    lines_df, demand_df, generation_availability_df, generation_df, value_of_lost_load, relaxation = read_input_data(base_path)
    # Building Ground Truth
    T = demand_df['time_step'].nunique()
    investment_df, line_flow_df, loss_of_load_df, production_df, output_scalars = read_output_data(base_path)

    # (Technology,Location) nodes
    tech_location = [(row['technology'], row['location']) for _, row in generation_df.drop_duplicates(subset=['technology', 'location']).iterrows()]

    tech_names = [f"{t}|{l}" for t, l in tech_location]
    tech_to_index = {name: i for i, name in enumerate(tech_names)}
    
    # Location nodes
    locations = sorted(demand_df['location'].unique())

    loc_to_index = {loc: i for i, loc in enumerate(locations)}
    
    # Demand nodes
    demand_nodes = [f"D_{loc}" for loc in locations]
    demand_to_index = {name: i for i, name in enumerate(demand_nodes)}

    technologies = generation_df['technology'].unique()

    # generation_df['availability'] = 1 # Change so it takes in these newer energies

    time_steps = range(1, T + 1)
    # location_time_combinations = generation_availability_df[['location', 'time_step']].drop_duplicates()

    location_time_combinations = pd.MultiIndex.from_product(
        [locations, time_steps],
        names=["location", "time_step"]
    ).to_frame(index=False)

    for tech in technologies:
        if tech not in TRADITIONAL_ENERGY:
            continue
        # Doing only for Traditional Energty
        # print(f"ADDING FOR TECH")

        row = location_time_combinations.copy()

        unique_loc = row['location'].unique()
        
        for loc in unique_loc: 
        
            if f"{tech}|{loc}" not in tech_to_index.keys():
                continue
            filtered_row = row[row['location'] == loc].copy()
            filtered_row['technology'] = tech
            filtered_row['availability'] = 1
            filtered_row = filtered_row[generation_availability_df.columns]



            # Append to the original DataFrame
            generation_availability_df = pd.concat([generation_availability_df, filtered_row], ignore_index=True)
 
    # print unique technology and location pairrs
    unique_pairs = generation_availability_df[['location', 'technology']].drop_duplicates()
    print(unique_pairs)

    expanded_generation_df = generation_availability_df.merge(
        generation_df,
        on=['technology', 'location'],
        how='left'  # Or 'inner' if you're certain they all exist
    )


    if use_investment_as_feature:
        expanded_generation_df = expanded_generation_df.merge(
            investment_df[['location', 'technology', 'units']].rename(columns={'units': 'investment_unit'}),
            on=['location', 'technology'],
            how='left'
        )
        expanded_generation_df['investment_unit'] = expanded_generation_df['investment_unit'].fillna(0)
        
    # print(expanded_generation_df.head())
    # print(expanded_generation_df.shape)

    # print(expanded_generation_df[expanded_generation_df['technology'] == 'WindOn'].head())

    # print(expanded_generation_df[expanded_generation_df['technology'] == 'Oil'].head())


    
    # Technology features
    expanded_generation_df = expanded_generation_df.sort_values(by=['time_step','technology','location']).reset_index(drop=True)
    


    if use_investment_as_feature:
        tech_features = expanded_generation_df[['variable_cost', 'unit_capacity', 'investment_unit' , 'availability']]
    else:
        tech_features = expanded_generation_df[['variable_cost', 'unit_capacity', 'availability']]

    
    
    tech_features = torch.tensor(tech_features.to_numpy(dtype=np.float32), dtype=torch.float32).reshape(T,len(tech_to_index), -1)

    # Location features
    loc_features = torch.zeros((len(locations), 1))  # or any features you have

    # Demand features
    # pivot demand_df so each row is a timestep, each col is a location
    pivoted = demand_df.pivot(index='time_step', columns='location', values='demand')
    demand_features = torch.tensor(pivoted.values, dtype=torch.float32)  # shape (T, L)
 

    

    # Edges: Technology -> Location
    tech2loc_edges = []

    for _, row in generation_df.iterrows():
        techloc = f"{row['technology']}|{row['location']}"
        tech_index = tech_to_index[techloc]
        loc_index = loc_to_index[row['location']]
        
        tech2loc_edges.append([tech_index, loc_index])

    # Edges: Location -> Demand
    loc2demand_edges = []
    for loc in locations:
        loc_idx = loc_to_index[loc]
        demand_idx = demand_to_index[f"D_{loc}"]
        loc2demand_edges.append([loc_idx, demand_idx])




    flow2loc_edges = []
    loc2flow_edges = []
    flow_node = {}
    flow_features = []
    count = 0

    flow_location_map = []

    for _, row in lines_df.iterrows():
        u, v = row['from'], row['to']
        if u in loc_to_index and v in loc_to_index:
            u_index = loc_to_index[u]
            v_index = loc_to_index[v]

            # Export Flow
            flow_name = f"{u}-{v}"
            flow_node[flow_name] = count
            # flow_features.append([row['export_capacity'],row['import_capacity']])
            flow_features.append([-row['import_capacity'],row['export_capacity']])
            flow_location_map.append((u_index,v_index))
            count += 1

            # Export u->v
            loc2flow_edges.append([u_index, flow_node[flow_name]]) # u->f
            loc2flow_edges.append([v_index, flow_node[flow_name]]) # v->f
            flow2loc_edges.append([flow_node[flow_name], u_index]) # f->u
            flow2loc_edges.append([flow_node[flow_name], v_index]) # f->v



    flow_features = torch.tensor(flow_features, dtype=torch.float) 
    
    tech2loc_index = torch.tensor(tech2loc_edges).t().contiguous()
    
    loc2flow_index = torch.tensor(loc2flow_edges).t().contiguous()
    flow2loc_index = torch.tensor(flow2loc_edges).t().contiguous()
    loc2demand_index = torch.tensor(loc2demand_edges).t().contiguous()



    # Production GT for each technology location pair
    production_gt = torch.zeros((T, len(tech_to_index), 1), dtype=torch.float)

    
    for _, row in generation_df.iterrows():
        loc = row['location']
        tech = row['technology']
        index = tech_to_index[f"{tech}|{loc}"]
        match_row = production_df[(production_df['location'] == loc) & (production_df['technology'] == tech)]  
        # print(f"Match Row: {match_row}")
        for time in range(T):

            if (time+1) not in match_row['time_step'].tolist() or match_row.empty:
                production_gt[time, index, 0] = 0.0
            else:
                production_gt[time, index, 0] = match_row.loc[match_row['time_step'] == (time+1), 'production'].values[0]


    p_loss_gt = torch.zeros((T, len(loc_to_index), 1), dtype=torch.float)

    for _, row in loss_of_load_df.iterrows():
        loc = row['location']
        time_step = int(row['time_step']) - 1  # 0-based index
        loss = float(row['loss_of_load'])

        if loc in loc_to_index:
            loc_idx = loc_to_index[loc]
            p_loss_gt[time_step, loc_idx, 0] = loss



    flow_gt = torch.zeros((T, len(flow_node), 1), dtype=torch.float)
    for _, row in line_flow_df.iterrows():


        export_key = f"{row['from']}-{row['to']}"
        # import_key = f"{row['to']}-{row['from']}"
        time_step = int(row['time_step']) - 1  
        flow = row['flow']
        # if export_key in flow_node:
        flow_gt[time_step, flow_node[export_key], 0] = flow

    node_feat = {
        'technology': tech_features,
        'location': loc_features,
        'demand': demand_features,
        'flow': flow_features
    }
    edge_feat = {
        'tech2loc': tech2loc_index,
        'loc2flow': loc2flow_index,
        'flow2loc': flow2loc_index,
        'loc2demand': loc2demand_index
    }
    gt = {
        'p_loss': p_loss_gt,
        'production': production_gt,
        'flow': flow_gt
    }
    scalars = {
        'loss_load': value_of_lost_load,
        'relaxation': relaxation,
        'total_investment_cost': output_scalars['total_investment_cost'],
        'total_operational_cost': output_scalars['total_operational_cost'],
    }
    if plot:
        plot_hetero_graph_from_data(node_feat, edge_feat, tech_to_index, loc_to_index, demand_to_index)
    return node_feat, edge_feat, gt, flow_location_map, scalars

    


def fully_connect(num_src, num_dst):
    src = torch.arange(num_src).repeat_interleave(num_dst)
    dst = torch.arange(num_dst).repeat(num_src)
    return torch.stack([src, dst], dim=0)

def build_graph_list(node_feat, edge_index, gt,total_time, topology):
    tech_features = node_feat['technology']
    loc_features = node_feat['location']
    demand_features = node_feat['demand']
    flow_features = node_feat['flow']


    tech2loc_index = edge_index['tech2loc']
    flow2loc_index = edge_index['flow2loc']
    loc2flow_index = edge_index['loc2flow']
    loc2demand_index = edge_index['loc2demand']

    production_gt = gt['production']
    flow_gt = gt['flow']
    p_loss_gt = gt['p_loss']

    # Start building graph
    graph_list = []

    if topology == FULLY_CONNECTED: 
        node_types = ['technology', 'location', 'demand', 'flow']

        for t in range(total_time):
            data = HeteroData()
            data['technology'].x = tech_features[t]
            data['technology'].num_nodes = tech_features[t].size(0)

            data['location'].x = loc_features
            data['location'].num_nodes = loc_features.size(0)

            data['demand'].x = demand_features[t].unsqueeze(1)
            data['demand'].num_nodes = demand_features[t].shape[0]

            data['flow'].x = flow_features
            data['flow'].num_nodes = flow_features.size(0)

            # ---- fully connect every (src_type, dst_type) pair ----
            num_nodes = {
                'technology': data['technology'].num_nodes,
                'location':   data['location'].num_nodes,
                'demand':     data['demand'].num_nodes,
                'flow':       data['flow'].num_nodes,
            }

            for src_type in node_types:
                for dst_type in node_types:
                    edge_idx = fully_connect(num_nodes[src_type], num_nodes[dst_type])

                    # Give each relation a unique, stable name
                    rel = f'{src_type}_to_{dst_type}'
                    data[(src_type, rel, dst_type)].edge_index = edge_idx

            # ---- targets (same as your original) ----
            data['technology'].y = production_gt[t]     # (num_tech_nodes, 1)
            data['flow'].y       = flow_gt[t]           # (num_flow_nodes, 1)
            data['location'].y   = p_loss_gt[t]         # (num_location_nodes, 1)

            graph_list.append(data)

    else:
        for t in range(total_time):
            data = HeteroData()
            data['technology'].x = tech_features[t]
            data['technology'].num_nodes = tech_features[t].size(0)
            data['location'].x = loc_features
            data['location'].num_nodes = loc_features.size(0)
            data['demand'].x = demand_features[t].unsqueeze(1)  # for current time t
            data['demand'].num_nodes = demand_features[t].shape[0]
            data['flow'].x = flow_features
            data['flow'].num_nodes = flow_features.size(0)

            data['technology', 'powers', 'location'].edge_index = tech2loc_index
            data['location', 'powered_by', 'technology'].edge_index = tech2loc_index[[1, 0]]  # reverse
            data['location', 'feeds', 'demand'].edge_index = loc2demand_index
            data['demand', 'fed_by', 'location'].edge_index = loc2demand_index[[1, 0]]       # reverse
            data['flow', 'connected_to', 'location'].edge_index = flow2loc_index
            data['location', 'connected_from', 'flow'].edge_index = loc2flow_index

            data['technology'].y = production_gt[t]  # shape: (num_tech_nodes, 1)
            data['flow'].y = flow_gt[t]              # shape: (num_flow_nodes, 1)
            data['location'].y = p_loss_gt[t]        # shape: (num_location_nodes, 1)
            graph_list.append(data)

    return graph_list

if __name__ == "__main__":
    BASE_PATH = "instances/4Nodes-ren-2-cycle"
    DIRECTED = True
    #G, vertex_to_name = build_graph(BASE_PATH, graph_save_path=SAVE_PATH, json_save_path=JSON_SAVE_PATH, directed=DIRECTED, plot = False, seperate_technology = True, save = False)
    #print(vertex_to_name)

    # G = torch.load("./SimpleGridGraph.pt",weights_only=False)
    # with open("index_to_name.json", "r") as f:
    #     vertex_to_name = json.load(f)
    # print(f"Vertex to Name: {vertex_to_name}")
    # node_feature, edge_feature = build_features(BASE_PATH,G, vertex_to_name)
    # print(f"Node Feature: {node_feature.shape} Edge Feature: {edge_feature.shape}")
    # node_gt, edge_gt = build_ground_truth(
    #     BASE_PATH, G, vertex_to_name
    # )

    # print(f"Node GT: {node_gt.shape} Edge GT: {edge_gt.shape}")
    # print(f"G edge index: {G.edge_index.shape}")

    # print(node_feature[0:2,:,:])

    node_feat,edge_feat, gt,_,scalars  = build_hetero_graph(BASE_PATH, use_investment_as_feature=True, plot = False)
    # print out all shape 
    print("--------------Output Summary--------------")
    print(f"Tech Features: {node_feat['technology'].shape}")
    print(f"Location Features: {node_feat['location'].shape}")
    print(f"Demand Features: {node_feat['demand'].shape}")
    print(f"Flow Features: {node_feat['flow'].shape}")
    print(f"Tech to Location Index: {edge_feat['tech2loc']}")
    print(f"Flow to Location Index: {edge_feat['flow2loc']}")
    print(f"Location to Flow Index: {edge_feat['loc2flow']}")
    print(f"Location to Demand Index: {edge_feat['loc2demand']}")
    print(f"Production GT: {gt['production'].shape}")
    print(f"Flow GT: {gt['flow'].shape}")

    # print(f"Loss Load: {scalars['loss_load']}")
    # print(f"Relaxation: {scalars['relaxation']}")
    print(f"Tech2Loc {edge_feat['tech2loc']}")

    # total_production_t0 = gt['production'][t, :, 0].sum()
    # total_demand_t0 = node_feat['demand'][t, :].sum()

    # print(f"Total production at t = 0: {total_production_t0.item():.4f}")
    # print(f"Total demand at t = 0: {total_demand_t0.item():.4f}")
    print(f"Flow Feat: {node_feat['flow']}")
    print(f"Flow GT: {gt['flow']}")


