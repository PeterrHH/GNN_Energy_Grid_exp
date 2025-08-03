import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

from GraphBuilder import build_hetero_graph
from Scalar_Hetero import HeteroGraphScalar
from Eval import calculate_loss, summarize_feasibility, calc_constraint_violation
from Report import Report

class ProportionalCompletionLayer(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, p_hat, p_max, total_demand):
        p_sig = torch.sigmoid(p_hat)
        p_bnd = p_max * p_sig
        S = p_bnd.sum()
        D = total_demand

        if torch.isclose(S, D, atol=1e-4):
            return p_bnd

        if S < D:
            shortfall = D - S
            slack = p_max - p_bnd
            total_slack = slack.sum() + self.eps
            if total_slack < shortfall:
                return p_max.clone()
            alpha = shortfall / total_slack
            p_out = p_bnd + alpha * slack
            return p_out
        else:
            surplus = S - D
            total_bnd = S + self.eps
            if total_bnd < self.eps:
                return torch.zeros_like(p_bnd)
            beta = surplus / total_bnd
            p_out = (1.0 - beta) * p_bnd
            return p_out


class FlowBoundLayer(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, flow, flow_capacity):
        import_cap = flow_capacity[:, 0].unsqueeze(1)
        export_cap = flow_capacity[:, 1].unsqueeze(1)
        bounded_flow = torch.where(
            flow < 0,
            torch.sigmoid(-flow) * import_cap,
            torch.sigmoid(flow) * export_cap
        )
        return bounded_flow


class MLPModel(torch.nn.Module):
    def __init__(self, hidden_dim=64, use_investment_as_feature=True, repair=True):
        super().__init__()
        self.repair = repair
        tech_input_dim = 4 if use_investment_as_feature else 3
        self.tech_mlp = nn.Sequential(
            nn.Linear(tech_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.flow_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        if repair:
            self.proportional_completion = ProportionalCompletionLayer()
            self.flow_bound = FlowBoundLayer()

    def forward(self, x_dict):
        tech_input = x_dict['technology']
        flow_input = x_dict['flow']
        demand_input = x_dict['demand']

        raw_prod = self.tech_mlp(tech_input)
        raw_flow = self.flow_mlp(flow_input)

        if self.repair:
            max_capacity = tech_input[:, 1] * tech_input[:, 2] * tech_input[:, 3]
            total_demand = demand_input[:, 0].sum()
            repaired_prod = self.proportional_completion(
                p_hat=raw_prod.squeeze(1),
                p_max=max_capacity,
                total_demand=total_demand
            ).unsqueeze(1)
            repaired_flow = self.flow_bound(raw_flow, flow_input)
        else:
            repaired_prod = raw_prod
            repaired_flow = raw_flow

        return repaired_prod, repaired_flow


def build_dataset(base_path, use_investment_as_feature):
    node_feat, edge_index, gt, flow_loc_mapping, scalars = build_hetero_graph(base_path, use_investment_as_feature)
    tech_feat, loc_feat, demand_feat, flow_feat = node_feat['technology'], node_feat['location'], node_feat['demand'], node_feat['flow']
    prod_gt, flow_gt = gt['production'], gt['flow']
    p_loss_gt = gt['p_loss']

    scalar = HeteroGraphScalar()
    tech_feat, loc_feat, demand_feat, flow_feat = scalar.normalize_node_features(tech_feat, loc_feat, demand_feat, flow_feat)
    prod_gt, flow_gt = scalar.normalize_node_gt(prod_gt, flow_gt)

    data_list = []
    for t in range(demand_feat.shape[0]):
        data = HeteroData()
        data['technology'].x = tech_feat[t]
        data['technology'].y = prod_gt[t]
        data['location'].x = loc_feat
        data['location'].y = p_loss_gt[t]
        data['demand'].x = demand_feat[t].unsqueeze(1)
        data['flow'].x = flow_feat
        data['flow'].y = flow_gt[t]
        data_list.append(data)

    return data_list, edge_index, scalar, flow_loc_mapping, scalars


def train_mlp(base_path, hidden_dim=64, learning_rate=0.01, epochs=100, batch_size=32,
              use_investment_as_feature=True, repair=True, save_model=True, save_path=".", model_name="MLPModel"):

    data_list, edge_index, scalar, _, _ = build_dataset(base_path, use_investment_as_feature)
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = MLPModel(hidden_dim=hidden_dim, use_investment_as_feature=use_investment_as_feature, repair=repair)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred_prod, pred_flow = model(batch.x_dict)
            batch_size = batch['technology'].batch.max().item() + 1
            prod_loss, flow_loss, flow_cap_loss, balance_loss,_ = calculate_loss(
                pred_prod, pred_flow, batch, batch_size,
                loc2flow=edge_index['loc2flow'], tech2loc=edge_index['tech2loc']
            )
            loss = prod_loss + flow_loss + flow_cap_loss + balance_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Train Loss = {total_loss:.4f}")

    if save_model:
        save_trained_model(model, save_path, model_name, {
            'hidden_dim': hidden_dim,
            'use_investment_as_feature': use_investment_as_feature,
            'repair': repair
        })

    return model, test_loader, scalar, edge_index


def save_trained_model(model, save_path, name, config):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    print(f"Model saved to {model_path}")


def load_and_evaluate_model(model_path, base_path,training_time):
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    print(f"Loaded model with config: {config}")

    model = MLPModel(
        hidden_dim=config['hidden_dim'],
        use_investment_as_feature=config['use_investment_as_feature'],
        repair=config['repair']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_list, edge_index, scalar, flow_loc_mapping, scalars = build_dataset(base_path, config['use_investment_as_feature'])
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    # test_loader = DataLoader(data_list, batch_size=1, shuffle=False)

    report = Report(loss_cost=scalars['loss_load'])
    report.training_time = training_time

    for idx, batch in enumerate(test_loader):
        pred_prod, pred_flow = model(batch.x_dict)
        pred_prod, pred_flow = scalar.inverse_transform(pred_prod, pred_flow)
        gt_prod, gt_flow = scalar.inverse_transform(batch['technology'].y, batch['flow'].y)
        tech_feat, demand_feat, flow_feat = scalar.inverse_data(batch)

        violation_summary = summarize_feasibility(
            demand=demand_feat,
            tech_feat=tech_feat,
            flow=flow_feat.squeeze(-1),
            pred_prod=pred_prod.squeeze(),
            pred_flow=pred_flow.squeeze(),
            print_summary=False
        )
        is_feasible = all(violation_summary.values())

        total_prod = []
        tech2loc_index = edge_index['tech2loc']
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
            # print(f"--- flow_as_first_index: {flow_as_first_index}, flow_as_second_index: {flow_as_second_index}, total_prod: {total_prod}")
            # print("--------------------------------")
            # Find production that belongs to this location loc_node,
            lhs = total_prod + flow_as_first_index + flow_as_second_index # minus incoming because if its import, production should be negative
            rhs = demand_feat[loc]
            e_i = rhs - lhs
            loss_of_load.append(e_i.item())
        loss_of_load = torch.tensor(loss_of_load).unsqueeze(1)  # -> shape (N_demand, 1)


        report.add_instance(
            instance_idx=idx,
            variable_cost=tech_feat[:, 0].detach().cpu().numpy(),
            loss_of_load=loss_of_load.squeeze().detach().cpu().numpy(),
            production=pred_prod.squeeze().detach().cpu().numpy(),
            production_gt=gt_prod.squeeze().detach().cpu().numpy(),
            flow=pred_flow.squeeze(-1).detach().cpu().numpy(),
            flow_gt=gt_flow.squeeze(-1).detach().cpu().numpy(),
            demand_feat=demand_feat.squeeze().detach().cpu().numpy(),
            feasible=is_feasible,
            p_loss_gt = loss_of_load.squeeze().detach().cpu().numpy()
        )

    report.make_report(
        save_path="evaluation_logs2",
        filename="mlp_summary.csv",
        node_filename="mlp_node_details.csv"
    )


def evaluate_mlp(model, loader, scalar, edge_index):
    model.eval()
    total_prod_loss = 0
    total_flow_loss = 0
    with torch.no_grad():
        for batch in loader:
            pred_prod, pred_flow = model(batch.x_dict)
            batch_size = batch['technology'].batch.max().item() + 1
            prod_loss, flow_loss, _, _,_ = calculate_loss(
                pred_prod, pred_flow, batch, batch_size,
                loc2flow=edge_index['loc2flow'],
                tech2loc=edge_index['tech2loc']
            )
            total_prod_loss += prod_loss.item()
            total_flow_loss += flow_loss.item()
    print(f"Test Production Loss: {total_prod_loss / len(loader):.4f}")
    print(f"Test Flow Loss: {total_flow_loss / len(loader):.4f}")





if __name__ == "__main__":
    base_path = "Instances/3Nodes-ren-no-cycle"  # Change to your instance path
    # calculate training time give me code
    import time
    now = time.time()

    model, test_loader, scalar, edge_index = train_mlp(
        base_path=base_path,
        hidden_dim=64,
        learning_rate=0.01,
        epochs=100,
        use_investment_as_feature=True,
        repair=True,
        save_model=True,
        save_path="../saved_models",
        model_name="MLPModel_small_instance3"
    )
    training_time = time.time() - now 

    # new_topology_path = "Instances/4Nodes-ren-no-cycle"
    new_topology_path = base_path
    load_and_evaluate_model("../saved_models/MLPModel_small_instance3.pt", new_topology_path, training_time)
