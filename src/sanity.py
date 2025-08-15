# sanity.py
import torch
from torch_scatter import scatter_add

from HeteroGNN import build_hetero_graph, build_graph_list
from Eval import calculate_loss
from Scalar_Hetero import HeteroGraphScalar
from utils import *

# ---------- Endpoints and caps ----------
def ends_from_flow2loc(flow2loc, num_flows: int) -> torch.Tensor:
    """flow2loc: [2, 2E] with pairs (flow_id, loc_id). Returns ends [E,2]=(A,B)."""
    buckets = [[] for _ in range(num_flows)]
    f_ids = flow2loc[0].tolist()
    l_ids = flow2loc[1].tolist()
    for f, l in zip(f_ids, l_ids):
        buckets[f].append(l)
    ends = []
    for f in range(num_flows):
        assert len(buckets[f]) == 2, f"Flow {f} connects {len(buckets[f])} locs: {buckets[f]}"
        A, B = buckets[f]  # canonical order
        ends.append((A, B))
    return torch.tensor(ends, dtype=torch.long)

def positive_caps(flow_feat: torch.Tensor) -> torch.Tensor:
    """Convert [-import, +export] → [cap_AB, cap_BA] = [abs(first), abs(second)]"""
    caps = torch.stack([flow_feat[:,0].abs(), flow_feat[:,1].abs()], dim=1)
    return caps

# ---------- Bounding & inflow ----------
def bound_flow_tanh(flow_raw: torch.Tensor, caps_pos: torch.Tensor) -> torch.Tensor:
    """
    flow_raw: [E,1] or [B,E,1]
    caps_pos: [E,2] = [cap_AB>=0, cap_BA>=0]
    Returns flow in [-cap_BA, +cap_AB].
    """
    def _one(f, caps):
        cap_ab = caps[:,0].unsqueeze(-1)  # [E,1]
        cap_ba = caps[:,1].unsqueeze(-1)
        z = torch.tanh(f)                 # (-1,1)
        pos = torch.relu(z)
        neg = -torch.relu(-z)
        return pos * cap_ab + neg * cap_ba  # [-cap_ba, +cap_ab]

    if flow_raw.dim() == 2:   # [E,1]
        return _one(flow_raw, caps_pos)
    elif flow_raw.dim() == 3: # [B,E,1]
        B, E, _ = flow_raw.shape
        outs = []
        for b in range(B):
            outs.append(_one(flow_raw[b], caps_pos))
        return torch.stack(outs, dim=0)
    else:
        raise ValueError("flow_raw must be [E,1] or [B,E,1]")

def net_inflow_from_flows(flow: torch.Tensor, ends: torch.Tensor, n_loc: int) -> torch.Tensor:
    """
    flow: [E,1] or [B,E,1] (already bounded/signed)
    ends: [E,2] with (A,B). Positive flow = A→B.
    returns: [N_loc] or [B,N_loc]
    """
    A, B = ends[:,0], ends[:,1]
    def _one(f):
        f = f.squeeze(-1)             # [E]
        idx = torch.cat([A, B], dim=0)
        val = torch.cat([-f, +f], dim=0)
        return scatter_add(val, idx, dim=0, dim_size=n_loc)

    if flow.dim() == 2:
        return _one(flow)             # [N_loc]
    elif flow.dim() == 3:
        outs = []
        for b in range(flow.size(0)):
            outs.append(_one(flow[b]))
        return torch.stack(outs, dim=0)  # [B,N_loc]
    else:
        raise ValueError("flow must be [E,1] or [B,E,1]")

# ---------- Production aggregation ----------
def loc_prod_from_tech(pred_prod: torch.Tensor, tech2loc: torch.Tensor, n_loc: int) -> torch.Tensor:
    """
    pred_prod: [M,1] or [B,M,1]
    tech2loc: [2, M]   (row0=tech_idx, row1=loc_idx)
    returns: [N_loc] or [B,N_loc]
    """
    rows = tech2loc[1]    # loc indices
    cols = tech2loc[0]    # tech indices
    M = cols.numel()
    import torch.nn.functional as F

    # Sparse one-hot mapping P: [N_loc, M] where P[loc, tech]=1 if tech->loc
    P = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.ones(M, dtype=torch.float, device=rows.device),
        size=(n_loc, M)
    ).to_dense()

    def _one(p):
        p = p.squeeze(-1)            # [M]
        return torch.einsum("nm,m->n", P, p)

    if pred_prod.dim() == 2:
        return _one(pred_prod)       # [N_loc]
    elif pred_prod.dim() == 3:
        outs = []
        for b in range(pred_prod.size(0)):
            outs.append(_one(pred_prod[b]))
        return torch.stack(outs, dim=0)  # [B,N_loc]
    else:
        raise ValueError("pred_prod must be [M,1] or [B,M,1]")

# ---------- Balance loss exactly matching eval ----------
def balance_loss_aligned(pred_prod, pred_flow, demand, flow2loc, tech2loc):
    """
    pred_prod: [B,M,1] or [M,1]
    pred_flow: [B,E,1] or [E,1]
    demand:    [B,N,1] or [N,1]
    flow2loc:  [2, 2E] (pairs of (flow,loc))
    tech2loc:  [2, M]
    """
    B = (pred_prod.dim() == 3)
    N = demand.shape[-2]
    E = pred_flow.shape[-2]

    ends = ends_from_flow2loc(flow2loc, num_flows=E).to(pred_flow.device)
    net = net_inflow_from_flows(pred_flow, ends, N)         # [B,N] or [N]
    prod = loc_prod_from_tech(pred_prod, tech2loc, N)       # [B,N] or [N]
    dem  = demand.squeeze(-1)                               # [B,N] or [N]
    resid = prod + net - dem                                # supply - demand
    return (resid**2).mean(), resid


# add to sanity.py
def run_static_checks(node_feat, edge_index):
    n_flow = node_feat['flow'].shape[0]
    ends = ends_from_flow2loc(edge_index['flow2loc'], num_flows=n_flow)
    assert ends.shape[1] == 2, "Each flow must connect exactly 2 locations."
    caps = positive_caps(node_feat['flow'])
    assert (caps >= 0).all(), "Caps must be non-negative magnitudes."
    print("Static checks passed: endpoints and positive caps.")

def debug_alignment(batch, edge_index):
    B = batch['technology'].batch.max().item() + 1
    N = batch['demand'].x.shape[0] // B
    E = batch['flow'].x.shape[0]   // B
    M = batch['technology'].x.shape[0] // B

    pred_prod = batch['technology'].y.view(B, M, 1) * 0  # zeros
    pred_flow = batch['flow'].y.view(B, E, 1) * 0        # zeros

    L_bal, resid = balance_loss_aligned(
        pred_prod, pred_flow, batch['demand'].x.view(B, N, 1),
        edge_index['flow2loc'], edge_index['tech2loc']
    )
    print("Alignment check — balance loss on zeros (≈ mean(demand^2)):", float(L_bal))


if __name__ == "__main__":
    from sanity import debug_alignment
    from HeteroGNN import build_hetero_graph, build_graph_list
    from Scalar_Hetero import HeteroGraphScalar
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    from HeteroGNN import FULLY_CONNECTED, PHYSICAL_CONNECTED

    base_path = "Instances/2Nodes-no-ren"
    node_feat, edge_index, gt, flow_loc_mapping,_ = build_hetero_graph(base_path, 
                                                                       use_investment_as_feature = True)


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
    graph_list = build_graph_list(scaled_node_feat, edge_index, scaled_gt, total_time, topology = FULLY_CONNECTED)
    metadata = graph_list[0].metadata()
    print(f"METADATA: {metadata}")
    test_data = graph_list[-1]
    graph_list = graph_list[:-1]
    
    train_graphs, test_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    first_batch = next(iter(train_loader))
    debug_alignment(first_batch, edge_index)


def comp_b_check_validate(base_path, hidden_channels, learning_rate,
        n_epochs = 200, n_layers = 5, loss_mask=False, 
        logging = False, use_investment_as_feature = False, 
        add_self_loop = True,use_const_violation_loss = True,
        repair = True, save_model = False, topology = FULLY_CONNECTED):
    print(f"BASE PATH: {base_path}")
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

    # scaler = HeteroGraphScalar(scale = True)
    # tech_features, loc_features, demand_features, flow_features = scaler.normalize_node_features(tech_features, loc_features, demand_features, flow_features)
    # print(f"BEfore scailing, productiong GT {production_gt} Flow {flow_gt}\n")
    # production_gt, flow_gt = scaler.normalize_node_gt(production_gt, flow_gt)
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
    print(f"Scaled tech_features shape: {tech_features[1,:,:]}, loc_features shape: {loc_features}, demand_features shape: {demand_features[1,:]}, flow_features shape: {flow_features}\n")
    print(f"Scaled production_gt shape: {production_gt[1,:,:]}, flow_gt shape: {flow_gt[1,:,:]}, p_loss_gt shape: {p_loss_gt[1,:,:]}\n")
    total_time = demand_features.shape[0]

    # Create a hetero graph
    graph_list = build_graph_list(scaled_node_feat, edge_index, scaled_gt, total_time, topology)
    metadata = graph_list[0].metadata()
    
    test_data = graph_list[-1]
    graph_list = graph_list[:-1]
    
    train_graphs, test_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle = False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle = False)
    

    batch_size = 1
    data_batch = next(iter(test_loader))
    gt_prod = data_batch['technology'].y
    gt_flow = data_batch['flow'].y
    print(f"----------DEMAND FEATURE shape {data_batch['demand'].x.shape} &&& {data_batch['demand'].x}----------\n")
    print(f"For Test Loader, GT Production shape: {gt_prod.shape} && {gt_prod}, GT Flow shape: {gt_flow.shape} && {gt_flow}\n")
    prod_loss, flow_loss, flow_cap_loss, balance_loss, sum_flow_loss = calculate_loss(gt_prod, gt_flow, data_batch, batch_size, 
                                                        loc2flow_index, tech2loc_index,
                                                        loss_mask=loss_mask)
    
    print(f"prod_loss: {prod_loss}, flow_loss: {flow_loss}, flow_cap_loss: {flow_cap_loss}, balance_loss: {balance_loss}, sum_flow_loss: {sum_flow_loss}")


    print("Loc2Flow:")
    print(loc2flow_index)
    print("Tech2Loc:")
    print(tech2loc_index)

if __name__ == "__main__":
    # Example usage
    base_path = "Instances/3Nodes-no-ren-no-cycle"
    TOPOLOGY = FULLY_CONNECTED  # or PHYSICAL_CONNECTED
    training_time = comp_b_check_validate(base_path,
         learning_rate=0.005,
         hidden_channels= 64,
         n_epochs = 10,
         n_layers = 3,
         loss_mask=False,
         logging=False,
         use_investment_as_feature = True,
         add_self_loop= True,
         use_const_violation_loss = True,
         repair = False,
         save_model = True,
         topology = TOPOLOGY) # FULLY_CONNECTED, PHYSICAL_CONNECTED 