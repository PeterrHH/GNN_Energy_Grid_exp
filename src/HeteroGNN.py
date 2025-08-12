import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Tanh
from typing import List, Tuple
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import json
import wandb
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, HeteroConv, GATConv, GraphConv
import time
import torch.nn as nn

from GraphBuilder import build_hetero_graph, build_graph_list
from utils import *
from Scalar_Hetero import HeteroGraphScalar
from Report import Report
from Eval import calculate_loss, calc_constraint_violation, summarize_feasibility


class FlowBoundLayer(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, flow: torch.Tensor, flow_capacity: torch.Tensor) -> torch.Tensor:
        """
        Bound the flow values based on the provided flow capacities.
        flow: Tensor of shape [N, 1] representing the raw flow predictions.
        flow_capacity: Tensor of shape [N, 2] where:
            - flow_capacity[:, 0] is the import capacity (negative values)
            - flow_capacity[:, 1] is the export capacity (positive values)
        Returns:
            Bounded flow tensor of shape [N, 1].
        """
        import_cap = flow_capacity[:, 0].unsqueeze(1)
        export_cap = flow_capacity[:, 1].unsqueeze(1)
        # Check flow sign and bound accordingly
        # If flow is negative, it should be bounded by import capacity (which is negative)
        # Else, it should be bounded by export capacity (which is positive)
        bounded_flow = torch.where(
            flow < 0,
            torch.sigmoid(-flow) * import_cap,  # Sigmoid scaling for negative flow
            torch.sigmoid(flow) * export_cap    # Sigmoid scaling for positive flow
        )
        return bounded_flow

class ProportionalCompletionLayer(torch.nn.Module):
    """
    Step‐by‐step “Proportional‐Slack” completion for single‐timestep economic dispatch:
       1) Bound each raw p̂_i into [0, p_max_i] via sigmoid‐scaling.
       2) If sum(p_bnd) != total_demand, either:
           • shortfall (sum < demand): push each p_i up toward its p_max in proportion to slack, or
           • surplus  (sum > demand): push each p_i down proportionally to its own current value.
       3) If total slack is insufficient, we fall back on saturating everyone at p_max or p_min=0.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self,
                p_hat: torch.Tensor,  # raw network outputs, shape [N]
                p_max: torch.Tensor,  # upper bounds for each generator, shape [N]
                total_demand: torch.Tensor  # scalar (0‐d tensor) = sum of all consumer demands
                ) -> torch.Tensor:
        # 1) Bound‐repair via sigmoid:
        #    p_bnd[i] = p_max[i] * sigmoid(p_hat[i]).
        p_sig = torch.sigmoid(p_hat)  # in (0,1), shape [N]
        p_bnd = p_max * p_sig  # shape [N]; now 0 ≤ p_bnd ≤ p_max

        # 2) Compute the sum and compare to total_demand
        S = p_bnd.sum()  # scalar
        D = total_demand  # also a scalar
        # print(f"Sum of p_bnd: {S}, Total Demand: {D}")
        # If already nearly feasible, just return p_bnd:
        if torch.isclose(S, D, atol=1e-4):
            return p_bnd

        # a) Handle the “shortfall” case: S < D
        if S < D:
            shortfall = D - S  # > 0
            # each unit’s “slack” toward its p_max:
            slack = p_max - p_bnd  # shape [N], ≥ 0
            total_slack = slack.sum() + self.eps  # add eps to avoid zero
            # If even saturating everyone to p_max < D, then fallback:
            if total_slack < shortfall:
                return p_max.clone()
            # else distribute shortfall in proportion to slack_i / total_slack:
            alpha = shortfall / total_slack  # scalar in (0,1)
            # new p_i = p_bnd[i] + alpha * slack[i]
            p_out = p_bnd + alpha * slack
            return p_out

        # b) Handle the “surplus” case: S > D
        else:
            surplus = S - D  # > 0
            total_bnd = S + self.eps  # same as sum(p_bnd)
            # If total_bnd is zero, we cannot reduce → return zeros (feasibility lost).
            if total_bnd < self.eps:
                return torch.zeros_like(p_bnd)
            # distribute surplus proportionally:
            beta = surplus / total_bnd  # scalar in (0,1)
            # new p_i = p_bnd[i] - beta * p_bnd[i] = (1 - beta) * p_bnd[i]
            p_out = (1.0 - beta) * p_bnd
            return p_out


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, 
                 num_layers = 5, add_self_loops=False, 
                 repair = True, dropout_rate = 0.1,
                 use_investment_as_feature = True):
        super().__init__()
        self.node_type, self.edge_type = metadata
        self.dropout_rate = dropout_rate
        # Define a conv for each edge type
        if use_investment_as_feature:
            tech_dim = 4
        else:
            tech_dim = 3
        self.input_dim = {
            'technology': tech_dim,
            'location': 1,
            'demand': 1,
            'flow': 2,
        }
        self.hidden_channels = hidden_channels
        self.add_self_loops = add_self_loops
        self.repair = repair
        self.num_layers = num_layers
        self.encoder = nn.ModuleDict({
                nt: nn.Sequential(
                    nn.Linear(self.input_dim[nt], hidden_channels),
                    nn.ReLU() if nt != 'flow' else nn.Tanh(),
                    nn.Dropout(self.dropout_rate)
                )
                for nt in self.node_type
            })

        def make_conv(in_hidden: bool):
            return HeteroConv({
                et: SAGEConv((-1, -1) if not in_hidden else (hidden_channels, hidden_channels),
                             hidden_channels)
                for et in self.edge_type
            }, aggr='sum')

        # self.convs_list = nn.ModuleList([
        #     HeteroConv({
        #     ('technology', 'powers', 'location'): self.make_subsequent_layers(),
        #     ('location', 'powered_by', 'technology'): self.make_subsequent_layers(),
        #     ('location', 'feeds', 'demand'): self.make_subsequent_layers(),
        #     ('demand', 'fed_by', 'location'): self.make_subsequent_layers(),
        #     ('flow', 'connected_to', 'location'): self.make_subsequent_layers(),
        #     ('location', 'connected_from', 'flow'): self.make_subsequent_layers(),
        # }, aggr='sum') for _ in range(num_layers)
        # ])

        self.convs_list = nn.ModuleList(
            [make_conv(in_hidden=False)] + [make_conv(in_hidden=True) for _ in range(num_layers - 1)]
        )



        # self.self_loop_weights = nn.ParameterDict({ 
        #     'technology': nn.Parameter(torch.tensor(1.0)),  
        #     'location': nn.Parameter(torch.tensor(1.0)),
        #     'demand': nn.Parameter(torch.tensor(1.0)),
        #     'flow': nn.Parameter(torch.tensor(1.0)),
        # })

        self.self_loop_weights = nn.ParameterDict({
            nt: nn.Parameter(torch.tensor(1.0)) for nt in self.node_type
        })

        self.proportional_completion = ProportionalCompletionLayer()
        self.flow_bound = FlowBoundLayer()

        self.demand_lin = nn.Linear(self.hidden_channels, 1)  # final regression or classification head
        self.flow_lin = nn.Sequential(
            nn.Linear(hidden_channels, 1),
        )

    def make_first_layer(self):
        return SAGEConv(-1, self.hidden_channels)

    def make_subsequent_layers(self):
        return SAGEConv(self.hidden_channels, self.hidden_channels)


    def forward(self, x_dict, edge_index_dict):
        
        x_in = x_dict.copy()
        x_dict = {
            node_type: self.encoder[node_type](x)
            for node_type, x in x_dict.items()
        }
        x_encoded = x_dict.copy()

    
            
        # Add weighted self loop to its encoded feature
        for i, conv in enumerate(self.convs_list):
            x_out = conv(x_dict, edge_index_dict)
            
            x_out = {k: x.relu() if k != 'flow' else x.tanh() for k, x in x_out.items()}
            if self.add_self_loops:
                x_dict = {
                    k: self.self_loop_weights[k]*x_encoded[k] + x_out[k]
                    for k,x in x_out.items()
                }
            else:
                x_dict = x_out
        
        production_out =  self.demand_lin(x_dict['technology'])

        # Apply flow bound layer
        flow_out = self.flow_lin(x_dict['flow'])  # Ensure flow_out is of shape [N, 1]


        # If we use repair layer, we MUST USE INVESTMENT AS FEATURE
        if self.repair:

            assert x_in['technology'].shape[1] == 4, "Repair layer requires technology input to have 4 features: [variable_cost, unit_capacity, investment, availability]"
            input_demand = x_in['demand']
            input_tech = x_in['technology'] # [variable_cost, unit_capacity, investment, availability]
            max_capcacity = input_tech[:,1] * input_tech[:,2] * input_tech[:, 3]
            total_demand = input_demand[:,0].sum()
            

            production_out = self.proportional_completion(p_hat = production_out.squeeze(1), 
                                        p_max = max_capcacity, 
                                        total_demand = total_demand).unsqueeze(1)  # Add back the feature dimension
            
            flow_out = self.flow_bound(flow_out, x_in['flow'])



        return production_out, flow_out
    

def evaluate(model, loader, scalar, loss_mask, 
             loc2flow, tech2loc, use_investment = False):
    '''
    Evaluate a model on a given data loader.
    '''
    model.eval()

    total_production_loss = 0
    total_flow_loss = 0
    total_flow_cap_loss = 0
    total_balance_loss = 0
    total_sum_flow_loss = 0
    with torch.no_grad():
        for batch in loader:
            out_prod, out_flow = model(batch.x_dict, batch.edge_index_dict)

            batch_size = batch['technology'].batch.max().item() + 1
            prod_loss, flow_loss, flow_cap_loss, balance_loss, sum_flow_loss = calculate_loss(out_prod, out_flow, batch, batch_size, 
                                                                 loc2flow, tech2loc,
                                                                 loss_mask=loss_mask)
            total_production_loss += prod_loss.item()
            total_flow_loss += flow_loss.item()
            total_flow_cap_loss += flow_cap_loss.item()
            total_balance_loss += balance_loss.item()
            total_sum_flow_loss += sum_flow_loss.item()

    production_loss = total_production_loss / len(loader)
    flow_loss = total_flow_loss / len(loader)
    flow_cap_loss = total_flow_cap_loss / len(loader)
    balance_loss = total_balance_loss / len(loader)
    sum_flow_loss = total_sum_flow_loss / len(loader)
    return production_loss, flow_loss, flow_cap_loss, balance_loss, sum_flow_loss



def save_trained_model(model, save_path, name, config):
    """
    Save the model state dictionary and configuration parameters.
    """
    model_path = f"{save_path}/{name}.pt"
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config  # dictionary of args like use_investment_as_feature, etc.
    }
    torch.save(save_dict, model_path)
    print(f"Model and config saved to {model_path}")


def main(base_path, hidden_channels, learning_rate,
        n_epochs = 200, n_layers = 5, loss_mask=False, 
        logging = False, use_investment_as_feature = False, 
        add_self_loop = True,use_const_violation_loss = True,
        repair = True, save_model = False, topology = FULLY_CONNECTED,
        save_path = ".", save_model_name = "SaveModel"):
    
    node_feat, edge_index, gt, flow_loc_mapping,_ = build_hetero_graph(base_path, use_investment_as_feature)


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

    scaler = HeteroGraphScalar(scale = True)
    tech_features, loc_features, demand_features, flow_features = scaler.normalize_node_features(tech_features, loc_features, demand_features, flow_features)

    production_gt, flow_gt = scaler.normalize_node_gt(production_gt, flow_gt)
    scaled_node_feat = {
        'technology': tech_features,
        'location': loc_features,
        'demand': demand_features,
        'flow': flow_features
    }

    scaled_gt = {
        'production': production_gt,
        'flow': flow_gt,
        'p_loss': p_loss_gt
    }
    
    total_time = demand_features.shape[0]

    # Create a hetero graph
    graph_list = build_graph_list(scaled_node_feat, edge_index, scaled_gt, total_time, topology)
    metadata = graph_list[0].metadata()
    print(f"METADATA: {metadata}")
    test_data = graph_list[-1]
    graph_list = graph_list[:-1]
    
    train_graphs, test_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)


    model = HeteroGNN(hidden_channels=hidden_channels, metadata = metadata,
                      num_layers=n_layers, repair = repair,
                      add_self_loops=add_self_loop,
                      use_investment_as_feature=use_investment_as_feature)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 1e-4)


    wandb.init(
        mode = "online" if logging else "disabled",
        project="PowerFlowGNN", 
        name="HeteroTrialRun", 
        config={
        "epochs": n_epochs,
        "learning_rate": learning_rate,
        "loss_mask": loss_mask,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "base_path": base_path,
        'use_investment_as_feature': use_investment_as_feature,
        'add_self_loop': add_self_loop,
        'repair': repair,
        "use_const_violation_loss": use_const_violation_loss,
    })

    training_start = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_prod_loss = 0
        total_flow_loss = 0
        total_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out_prod, out_flow = model(batch.x_dict, batch.edge_index_dict)

            
            batch_size = batch['technology'].batch.max().item() + 1
            
            prod_loss, flow_loss, flow_cap_loss, balance_loss, total_loss = calculate_loss(out_prod, out_flow, batch, batch_size =batch_size,
                                                                 loc2flow = loc2flow_index,
                                                                 tech2loc = tech2loc_index)
            
            if use_const_violation_loss:
                # FLow
                loss = prod_loss + flow_loss + flow_cap_loss + balance_loss
            else:
                loss = prod_loss + flow_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_prod_loss += prod_loss.item()
            total_flow_loss += flow_loss.item()
            total_batches += 1
        ave_loss = total_loss / total_batches
        ave_prod_loss = total_prod_loss / total_batches
        ave_flow_loss = total_flow_loss / total_batches

        eval_prod_loss, eval_flow_loss, eval_flowcap_loss, eval_balance_loss,eval_sum_flow_loss = evaluate(model, val_loader, 
                                                                                scaler, loss_mask=loss_mask, 
                                                                                loc2flow = loc2flow_index,
                                                                                tech2loc = tech2loc_index,
                                                                                use_investment=use_investment_as_feature)
        if use_const_violation_loss:
            total_eval_loss = eval_prod_loss + eval_flow_loss + eval_flowcap_loss*10 + eval_balance_loss
        else:
            total_eval_loss = eval_prod_loss + eval_flow_loss
        wandb.log({
            'eval_prod_loss': eval_prod_loss,
            'eval_flow_loss': eval_flow_loss,
            'eval_loss': total_eval_loss,
            'train_loss': ave_loss,
            'train_prod_loss': ave_prod_loss,
            'train_flow_loss': ave_flow_loss,
        }, step=epoch)
        print(f'Epoch {epoch}, Train total loss: {ave_loss:.4f} Eval Prod Loss: {eval_prod_loss:.4f} Flow Loss: {eval_flow_loss:.4} Balance: {eval_balance_loss:.4f} Flow Cap Loss: {eval_flowcap_loss:.4f}')
    training_end = time.time()

    model.eval()
    
    start_time = time.time()
    production, flow = model(test_data.x_dict, test_data.edge_index_dict)
    inference_duration = time.time() - start_time
    print(f"Inference Time: {inference_duration:.6f} seconds out prod shape: {production.shape}, out flow shape: {flow.shape}")
    

    production, flow = scaler.inverse_transform(production, flow)
    production_gt, flow_gt = scaler.inverse_transform(test_data['technology'].y, test_data['flow'].y)
    tech_feat, demand_feat, flow_feat = scaler.inverse_data(test_data)
    #feat shape: {demand_feat.shape}, Tech feat shape: {tech_feat.shape}, Flow feat shape: {flow_feat.shape}")
    num_prod_tech = production.shape[0]
    num_flow_nodes = flow.shape[0]

    # ['variable_cost', 'unit_capacity', 'investment_unit' , 'availability']
    if use_investment_as_feature:
        invest = tech_feat[:, 2]
    availability = tech_feat[:, 3]
    unit_capacity = tech_feat[:, 1]
 
    print("------------------------------")
    for i in range(num_prod_tech):
        if use_investment_as_feature:
            full_capacity = availability[i] * invest[i] * unit_capacity[i]
        else:
            full_capacity = availability[i] * unit_capacity[i]
        print(f"Tech Node {i} --- Predicted: {production[i].item()} | GT: {production_gt[i].item()} | Unit Cap: {full_capacity.item()}")
    print("------------------------------")
    for i in range(num_flow_nodes):
        print(f"Flow Node {i} --- Predicted: {flow[i].item()} | GT: {flow_gt[i].item()} | Constraint ({flow_feat[i][0]},{flow_feat[i][1]})")
    print("------------------------------")

        

    test_prod_loss, test_flow_loss, test_flow_cap_loss, test_balance_loss,_ = evaluate(model, test_loader, scaler, 
                                                                            loc2flow = edge_index['loc2flow'], 
                                                                            tech2loc = edge_index['tech2loc'],
                                                                            loss_mask=loss_mask, 
                                                                            use_investment=use_investment_as_feature)
    # print("── Testset Performance ──")
    # print(f'Test Production Loss: {test_prod_loss:.4f} Flow Loss: {test_flow_loss:.4f}\n')

    if save_model:
        save_config = {
            'learning_rate': learning_rate,
            'hidden_channels': hidden_channels,
            'n_epochs': n_epochs,
            'n_layers': n_layers,
            'loss_mask': loss_mask,
            'logging': logging,
            'use_investment_as_feature': use_investment_as_feature,
            'add_self_loop': add_self_loop,
            'repair': repair,
            'dataset': base_path,
        }
        save_trained_model(model, save_path, save_model_name, save_config)

    return training_end - training_start


def evaluate_model(base_path, model_path, training_time,repair, topology):
    print("--------------------------------")
    print(f"Runnin evaluation using model {model_path} on dataset {base_path}")

    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    print(f"Model trained on {config['dataset']}")
    state_dict = checkpoint['model_state_dict']
    # load data
    node_feat, edge_index, gt, flow_loc_mapping,scalars = build_hetero_graph(base_path, config['use_investment_as_feature'])

    tech_features = node_feat['technology']
    loc_features = node_feat['location']
    demand_features = node_feat['demand']
    flow_features = node_feat['flow']
    tech2loc_index = edge_index['tech2loc']
    flow2loc_index = edge_index['flow2loc']
    loc2flow_index = edge_index['loc2flow']
    loc2demand_index = edge_index['loc2demand']

    production_gt_raw = gt['production']
    flow_gt_raw = gt['flow']
    p_loss_gt = gt['p_loss']

    # Normalize
    scaler = HeteroGraphScalar(scale=True)
    
    tech_features, loc_features, demand_features, flow_features = scaler.normalize_node_features(
        tech_features, loc_features, demand_features, flow_features
    )
    
    production_gt, flow_gt = scaler.normalize_node_gt(production_gt_raw, flow_gt_raw)
    print(f"prod before: {production_gt_raw[10]} prod after: {production_gt[10]}")
    print(f"flow before: {flow_gt_raw[10]} flow after: {flow_gt[10]}")
    total_time = demand_features.shape[0]
    scaled_node_feat = {
        'technology': tech_features,
        'location': loc_features,
        'demand': demand_features,
        'flow': flow_features
    }

    scaled_gt = {
        'production': production_gt,
        'flow': flow_gt,
        'p_loss': p_loss_gt
    }
    
    graph_list = build_graph_list(scaled_node_feat, edge_index, scaled_gt, total_time, topology)
    # graph_list = []

    # for t in range(total_time):
    #     data = HeteroData()
    #     data['technology'].x = tech_features[t]
    #     data['technology'].num_nodes = tech_features[t].size(0)
    #     data['location'].x = loc_features
    #     data['location'].num_nodes = loc_features.size(0)
    #     data['demand'].x = demand_features[t].unsqueeze(1)
    #     data['demand'].num_nodes = demand_features[t].shape[0]
    #     data['flow'].x = flow_features
    #     data['flow'].num_nodes = flow_features.size(0)

    #     data['technology', 'powers', 'location'].edge_index = tech2loc_index
    #     data['location', 'powered_by', 'technology'].edge_index = tech2loc_index[[1, 0]]
    #     data['location', 'feeds', 'demand'].edge_index = loc2demand_index
    #     data['demand', 'fed_by', 'location'].edge_index = loc2demand_index[[1, 0]]
    #     data['flow', 'connected_to', 'location'].edge_index = flow2loc_index
    #     data['location', 'connected_from', 'flow'].edge_index = loc2flow_index

    #     data['technology'].y = production_gt[t]
    #     data['flow'].y = flow_gt[t]
    #     data['location'].y = p_loss_gt[t]
    #     graph_list.append(data)
  
    metadata = graph_list[0].metadata()
    model = HeteroGNN(
        hidden_channels=config['hidden_channels'],
        metadata = metadata,
        num_layers=config['n_layers'],
        add_self_loops=config['add_self_loop'],
        repair=config['repair'],
        use_investment_as_feature=config['use_investment_as_feature']
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    model.repair = repair

    train_data, test_data = train_test_split(graph_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    # Create the Report object
    report = Report(loss_cost=scalars["loss_load"])
    report.inference_time = 0
    report.training_time = training_time
    failure_counts = {
            "p_below_zero":            0,
            "p_above_max":             0,
            "sum_balance_violation":   0,
            "exceed_export_capacity":  0,
            "exceed_import_capacity":  0,
        }
    # Evaluate over each time step individually and log each to report
    for idx, graph in enumerate(test_loader):
        # Run model inference
        start_time = time.perf_counter()
        pred_prod, pred_flow = model(graph.x_dict, graph.edge_index_dict)
        inference_duration = time.perf_counter() - start_time

        # Inverse scale predictions and ground truth
        pred_prod, pred_flow = scaler.inverse_transform(pred_prod, pred_flow)
        gt_prod, gt_flow = scaler.inverse_transform(graph['technology'].y, graph['flow'].y)


        tech_feat, demand_feat, flow_feat = scaler.inverse_data(graph)
        
        # Use the reshaped batch-style checker
        violation_summary = summarize_feasibility(
            demand=demand_feat,  # shape: [ N_demand, 1]
            tech_feat=tech_feat,  # shape: [N_tech, 4]
            flow=flow_feat.squeeze(-1),  # shape: [1, N_edges]
            pred_prod=pred_prod.squeeze(-1),  # shape: [N_tech, 1]
            pred_flow=pred_flow.squeeze(-1),  # shape: [N_edges, 1]
            print_summary=True,  # Set to True to print the summary
        )

        is_feasible = all(violation_summary.values())
        
        for check, passed in violation_summary.items():
            if not passed:
                failure_counts[check] += 1
        total_prod = []

        for loc_node in range(demand_feat.shape[0]):
            prod_index = (tech2loc_index[1] == loc_node).nonzero(as_tuple=True)[0]
            total_prod.append(pred_prod[prod_index].sum())


        loss_of_load = []
        for loc in range(demand_feat.shape[0]):
            incoming_index = [i for i, (a, b) in enumerate(flow_loc_mapping) if a == loc]
            outgoing_index = [i for i, (a, b) in enumerate(flow_loc_mapping) if b == loc]
            prod_index = (tech2loc_index[1] == loc).nonzero(as_tuple=True)[0]
            #print(f"For loc {loc} incoming {incoming_index} outgoing {outgoing_index} prod index {prod_index}")
            flow_as_first_index = - pred_flow[incoming_index].sum() if incoming_index else 0.0
            flow_as_second_index = pred_flow[outgoing_index].sum() if outgoing_index else 0.0
            total_prod = pred_prod[prod_index].sum()
            # Find production that belongs to this location loc_node,
            lhs = total_prod + flow_as_first_index + flow_as_second_index # minus incoming because if its import, production should be negative
            rhs = demand_feat[loc]
            e_i = rhs - lhs
            loss_of_load.append(e_i.item())
        loss_of_load = torch.tensor(loss_of_load).unsqueeze(1) 

            
        # remove all singleton dimensions and move to numpy
        prod = pred_prod.squeeze(-1).detach().cpu().numpy()  # -> shape (N_tech,)
        prod_gt = gt_prod.squeeze(-1).detach().cpu().numpy()  # -> shape (N_tech,)
        flow = pred_flow.squeeze(-1).detach().cpu().numpy()  # -> shape (N_flow,)
        flow_gt = gt_flow.squeeze(-1).detach().cpu().numpy()  # -> shape (N_flow,)
        cost = tech_feat[:, 0].detach().cpu().numpy()  # -> shape (N_tech,)
        gt_p_loss = graph['location'].y
        
        report.add_instance(
            instance_idx=idx,
            variable_cost=cost,
            loss_of_load=loss_of_load.squeeze().detach().cpu().numpy(),
            production=prod,
            production_gt=prod_gt,
            flow=flow,
            flow_gt=flow_gt,
            p_loss_gt = gt_p_loss.squeeze().detach().cpu().numpy(),
            demand_feat = demand_feat.squeeze().detach().cpu().numpy(),
            feasible=is_feasible
        )



        # Accumulate time
        report.inference_time += inference_duration

    total_instances = len(test_loader)
    print("\n=== Feasibility‐check failure summary ===")
    for check, count in failure_counts.items():
        pct = count / total_instances * 100
        print(f"  • {check:25s} failed in {count}/{total_instances} instances ({pct:.1f}%)")

    # Average inference time
    report.inference_time /= len(test_loader)
    print(f"Total Inference Time: {report.inference_time:.4f} seconds")
    # Save report
    report.make_report(
        save_path="evaluation_logs",
        filename="summary.csv",
        node_filename="node_details.csv",
        scalars = scalars
    )


    # print("\n-------------Feasibility Check Example------------------")
    # full_data_loader = DataLoader(graph_list, batch_size=len(graph_list))  # Single batch for test data
    # batch = next(iter(full_data_loader))
    # out_prod, out_flow = model(batch.x_dict, batch.edge_index_dict)

    # Total_time = production_gt.shape[0]
    # out_prod = out_prod.reshape(Total_time, -1)  # Reshape to [T, N_tech]
    # out_flow = out_flow.reshape(Total_time, -1)  # Reshape to

    # out_prod, out_flow = scaler.inverse_transform(out_prod, out_flow)

    # # Reshape into [T, N_tech] and [T, N_flow]
    # out_prod = out_prod.reshape(Total_time, -1)  
    # out_flow = out_flow.reshape(Total_time, -1) 

    # tech_features, demand_features, flow_features = scaler.inverse_batch_data(batch, T = test_loader.batch_size)
    # print(f"Tech features shape: {tech_features.shape}, Demand features shape: {demand_features.shape}, Flow features shape: {flow_features.shape}")
    # # Rescale Tech feature and flow feature
    # calc_constraint_violation(
    #     demand_features, 
    #     tech_features, 
    #     flow_features, 
    #     out_prod, 
    #     out_flow,
    #     flow_loc_mapping,
    #     tech2loc_index,
    #     tol=1e-4
    # )


if __name__ == "__main__":
    '''
    Set logging to False, to not log anything to wandb, only show these logs in the terminal locally.
    '''

    base_path = "Instances/2Nodes-no-ren"
    TOPOLOGY = FULLY_CONNECTED
    training_time = main(base_path,
         learning_rate=0.01,
         hidden_channels= 64,
         n_epochs = 20,
         n_layers = 3,
         loss_mask=False,
         logging=False,
         use_investment_as_feature = True,
         add_self_loop= True,
         use_const_violation_loss = True,
         repair = False,
         save_model = True,
         topology = TOPOLOGY, # FULLY_CONNECTED, PHYSICAL_CONNECTED 
         save_path= "../",
         save_model_name = "GNNModel-3Nodes-ren")

    # base_path = "Instances/4Nodes-ren-1-cycle"
    evaluate_model(base_path, "../GNNModel-3Nodes-ren.pt", training_time, repair = True, topology = TOPOLOGY)