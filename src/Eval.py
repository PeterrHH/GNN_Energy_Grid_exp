from GraphBuilder import build_hetero_graph
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import torch
from typing import List, Tuple


def build_incidence(loc2flow, num_nodes):
    '''
    Build Sparse Incidence Matrix from location to flow mapping
    '''
    rows, cols, vals = [], [], []
    u_list, v_list = loc2flow[0], loc2flow[1]
    for e in range(loc2flow.shape[1]):
        u, v = u_list[e].item(), v_list[e].item()
        rows.extend([u, v])
        cols.extend([e, e])
        vals.extend([-1.0, 1.0])  # u is source (+), v is sink (-). Since import is negative, multiply by -1.0
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float)
    return torch.sparse_coo_tensor(indices, values, size=(num_nodes, loc2flow.shape[1]))

def build_prod_mapping(tech2loc, num_nodes, num_techs):
    '''
    Build sparse mapping from technology to location
    '''
    rows = tech2loc[1].tolist()  # location indices
    cols = tech2loc[0].tolist()  # tech indices
    vals = [1.0] * len(rows)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float)
    return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_techs))



def build_edge_mapping_from_loc2flow(loc2flow, num_flow_nodes):

    edge_to_flow_index = []
    edge_sign = []

    # Build mapping: flow_node → [connected location indices]
    flow_to_locs = {f: [] for f in range(num_flow_nodes+1)}
    
    for loc, f in zip(loc2flow[0].tolist(), loc2flow[1].tolist()):

        flow_to_locs[f].append(loc)

    for i in range(loc2flow.shape[1]):
        loc = loc2flow[0, i].item()
        f = loc2flow[1, i].item()
        loc_pair = flow_to_locs[f]
        if len(loc_pair) != 2:
            raise ValueError(f"Flow node {f} does not connect exactly two locations: {loc_pair}")

        u, v = loc_pair  # use consistent ordering
        sign = +1.0 if loc == u else -1.0
        edge_to_flow_index.append(f)
        edge_sign.append(sign)

    return (
        torch.tensor(edge_to_flow_index, dtype=torch.long),
        torch.tensor(edge_sign, dtype=torch.float)
    )


def compute_balance_loss(demand,technology,flow,loc2flow,tech2loc, pred_prod, pred_flow, batch_size):
    '''
    Compute Constraint Violation Loss
    '''
    
    N =  demand.shape[0] // batch_size
    M = technology.shape[0] // batch_size
    F = flow.shape[0] // batch_size

    demand = demand.view(batch_size,N)
    technology = technology.view(batch_size, M ,-1)
    flow = flow.view(batch_size, F, -1)[0]

    pred_flow = pred_flow.view(batch_size, F)
    pred_prod = pred_prod.view(batch_size, M)


    # print(f"-----------Test things ------------------")
    loc_count = demand.shape[1]    
    tech_count = technology.shape[1]
    flow_count = F

    prod = pred_prod  # shape (T*|N|, 1)
    demand = demand # shape (T(|N|, 1)
    
    edge_to_flow_index, edge_sign = build_edge_mapping_from_loc2flow(loc2flow, flow_count)
    
    flow_expanded = pred_flow[:,edge_to_flow_index] * edge_sign  # [B, 6]

    

    A = build_incidence(loc2flow, num_nodes=loc_count)  # shape [3, 3]

    # print(f"TECH2LOC shape {tech2loc.shape} loc_count {loc_count} tech_count {tech_count}")
    P = build_prod_mapping(tech2loc, num_nodes=loc_count, num_techs=tech_count)


    A_dense = A.to_dense()         # shape: [N, 6]
    P_dense = P.to_dense()         # shape: [N, M]

    # Step 1: Compute net inflow at each location
    net_inflow = torch.einsum("ne,be->bn", A_dense, flow_expanded)  # [B, N]

    # Step 2: Compute local production at each location
    loc_prod = torch.einsum("nm,bm->bn", P_dense, prod.squeeze())             # [B, N]
    # print(f"loc_prod shape {loc_prod.shape} net_inflow shape {net_inflow.shape} demand shape {demand.shape}")
    
    # Step 3: Total supply = production + net inflow
    total_supply = loc_prod + net_inflow                            # [B, N]

    # Step 4: Compute violation w.r.t. demand
    # violation = (total_supply - demand) ** 2                      # [B, N]

    # # Step 5: Final loss
    # loss_constraint = violation.mean()

    # loss_constraint = (total_supply)
    # Alternative approach to compute oversupply
    oversupply = torch.clamp(total_supply - demand, min = 0.0)  # [B, N], only positive deviations
    loss_constraint = (oversupply ** 2).mean()

    return loss_constraint


def calculate_loss(pred_prod, pred_flow, batch, batch_size, loc2flow, tech2loc, loss_mask=False):
    '''
    batch['technology'].y, batch['flow'].y, 
    batch['flow'].x, batch['demand'].x,batch['technology'].x,
    batch['location', 'connected_from', 'flow'].edge_index,
    batch['technology', 'powers', 'location'].edge_index
    '''
    gt_prod = batch['technology'].y
    gt_flow = batch['flow'].y

    flow_feat = batch['flow'].x
    demand_feat = batch['demand'].x
    tech_feat = batch['technology'].x


    if loss_mask:
        mask = (gt_prod != 0).float()
        
        prod_loss = F.mse_loss(pred_prod * mask, gt_prod * mask, reduction='sum')
        prod_loss = prod_loss / (mask.sum() + 1e-8)  # Normalize only over non-zero targets
    else:

        prod_loss = F.mse_loss(pred_prod, gt_prod, reduction='mean')

    # Capacity
    import_capacity = flow_feat[:, 0]  # import capacity
    export_capacity = flow_feat[:, 1]  # export capacity
    
    # Calculate violation amounts
    import_violation = torch.clamp(import_capacity - pred_flow, min=0.0)  # Too negative
    export_violation = torch.clamp(pred_flow - export_capacity, min=0.0)  # Too positive

    # Total violation penalty (can be scaled if needed)
    violation_loss = ((import_violation + export_violation)**2).mean()

    flow_loss = F.mse_loss(pred_flow, gt_flow)

    total_flow = pred_flow.abs().sum()
    
    balance_loss = compute_balance_loss(demand_feat,tech_feat, flow_feat, loc2flow, tech2loc, pred_prod, pred_flow, batch_size)

    return prod_loss, flow_loss, violation_loss, balance_loss, total_flow



def calc_constraint_violation(demand: torch.Tensor,             # demand Faeture
                              tech_feat: torch.Tensor,          # Tech feature
                              flow: torch.Tensor,               # Flow feature
                              pred_prod: torch.Tensor,
                              pred_flow: torch.Tensor,
                              flow_loc_mapping:List[Tuple[int, int]],
                              tech_loc_mapping: torch.Tensor, # [2, N_tech], first row is tech index, second row is location index
                              tol: float = 1e-5):
    """
    Functiion to Calculate amount of Constraint Violation for a batch of data.

    RH is right hand side, LH is left hand side.›
    A simplified checker for single‐timestep ED: 
    Six inequality constraint
    - -p_i <= 0
    - p_i <= p_max
    - f_e <= capacity_export
    - -f_e <= capacity_import (Here we store capacity improt as negative, so f_e < capacity_import )
    - -e_i <= 0
    - e_i <= D_i
    Equality Constraint:
    - p_i - f + e = D

    For inequality constraints:
    (|LH - RH|/|RH|)

    Equality constraint
    rho =
    - Case 1: if RH has bound, rho = upper bound - lower bound
    - Case 2: if RH has no bound, rho = |RH| <- absolute value of right hand side

    returns;
    - eq_mean =  mean of (|LH - RH|/|RH|) over the eqaulity constarints across all instances
    - eq_max =  maximum of (|LH - RH|/|RH|) of th equality constraint for each instances, Then take the mean over all instances
    - ieq_mean = mean of (LH--RH)/rho over the inequality constraints across all instances
    - ieq_max = max of (LH--RH)/rho over the equality constraints for each instances, then take the mean across all instances
    """
    
    if tech_feat.shape[2] != 4: 
        print("Check is only valid with investment info")
        return

    T = tech_feat.shape[0]

    # Feature are [Var_cost, Unit_Cap, Investmnet, Availability]
    investment = tech_feat[:, :, 2]     # (T, N_tech)
    availability = tech_feat[:, :, 3]   # (T, N_tech)
    unit_capacity = tech_feat[:, :, 1]  # (T, N_tech)
    p_max = availability * investment * unit_capacity         # if not a generator, p_max = 0

    eq_violations = []
    ieq_violations = []
    
    N = demand.shape[1]  # Number of technology nodes
    equality_gap = torch.zeros(T,N)

    e_demand_percentage = torch.zeros(T, N)  # Store e_i / D_i for each location

    for t in range(T):

        # Need to see if can be optimised
        e_vec = torch.zeros(N, device=demand.device)
        for loc_node in range(demand.shape[1]):

            incoming_index = [i for i, (a, b) in enumerate(flow_loc_mapping) if a == loc_node]
            outgoing_index = [i for i, (a, b) in enumerate(flow_loc_mapping) if b == loc_node]
            prod_index = (tech_loc_mapping[1] == loc_node).nonzero(as_tuple=True)[0]

            flow_as_first_index = - pred_flow[t, incoming_index].sum() if incoming_index else 0.0
            flow_as_second_index = pred_flow[t, outgoing_index].sum() if outgoing_index else 0.0
            total_prod = pred_prod[t,prod_index].sum()

            # Find production that belongs to this location loc_node,
            lhs = total_prod + flow_as_first_index + flow_as_second_index # minus incoming because if its import, production should be negative
            rhs = demand[t, loc_node]
            e_i = rhs - lhs
            e_vec[loc_node] = e_i

            lhs += e_i
            gap =  ((lhs - rhs).abs() / (rhs.abs() + tol))
            # How to proceed
            equality_gap[t, loc_node] = gap.item()
        
        
        # ---------Inequality Constraints--------
        # 1. -p_i <= 0 <-> p_i >0
        # 2. p_i <= p_max
        rho_prod = p_max[t]
        mask = rho_prod > 0

        numerator_1 = torch.clamp(pred_prod[t].squeeze(), max=0).abs()
        numerator_2 = torch.clamp(pred_prod[t].squeeze() - p_max[t], min=0)
        
        ieq1 = torch.zeros_like(rho_prod)
        ieq2 = torch.zeros_like(rho_prod)

        ieq1[mask] = numerator_1[mask] / rho_prod[mask]
        ieq2[mask] = numerator_2[mask] / rho_prod[mask]

        # 3. f_e <= export_capcity
        # 4, -f_e <= -import_capacity in our case
        export_cap = flow[:, 1]             
        import_cap = flow[:, 0]             
        rho_flow = export_cap + import_cap.abs()  

        pred = pred_flow[t].squeeze()      

        numerator_ieq3 = torch.clamp(pred - export_cap, min=0)               # flow > export
        numerator_ieq4 = torch.clamp(pred.abs() - import_cap.abs(), min=0)   # |flow| > |import|

        mask = rho_flow > 0

        ieq3 = torch.zeros_like(rho_flow)
        ieq4 = torch.zeros_like(rho_flow)

        ieq3[mask] = numerator_ieq3[mask] / rho_flow[mask]
        ieq4[mask] = numerator_ieq4[mask] / rho_flow[mask]


        # 5 & 6. Extra-energy (storage) bounds: 0 <= e_i <= D_i
        rho_e = demand[t]                
        numerator_ieq5 = torch.clamp(-e_vec, min=0)           # e_i < 0
        numerator_ieq6 = torch.clamp(e_vec - rho_e, min=0)    # e_i > demand

        # Safe mask
        mask = rho_e > 0

        # Allocate result tensors
        ieq5 = torch.zeros_like(rho_e)
        ieq6 = torch.zeros_like(rho_e)

        # Safe division
        ieq5[mask] = numerator_ieq5[mask] / rho_e[mask]
        ieq6[mask] = numerator_ieq6[mask] / rho_e[mask]

        #print(f"e_vec {e_vec} demand {rho_e} | ieq5: {ieq5.max().item()}, ieq6: {ieq6.max().item()}")
        ieq_all = torch.cat([ieq1, ieq2, ieq3, ieq4, ieq5, ieq6])
        ieq_violations.append(ieq_all)
        # print(f"[Debug] Max in ieq1 (prod < 0): {ieq1.max().item()}")
        # print(f"[Debug] Max in ieq2 (prod > p_max): {ieq2.max().item()}")
        # print(f"[Debug] Max in ieq3 (flow > export capacity): {ieq3.max().item()}")
        # print(f"[Debug] Max in ieq4 (flow < import capacity): {ieq4.max().item()}")
        # print(f"[Debug] Max in ieq5 (e < 0): {ieq5.max().item()}")
        # print(f"[Debug] Max in ieq6 (e > demand): {ieq6.max().item()}")
        # print("#################################\n")


    

    # Compute final metric
    eq_mean = equality_gap.mean().item()
    eq_max = equality_gap.max(dim=1).values.mean().item()

    ieq_violations = torch.stack(ieq_violations)  
    ieq_mean = ieq_violations.mean().item()
   
    ieq_max = ieq_violations.max(dim=1).values.mean().item()
    print(f"Equality Mean: {eq_mean:.4f}, Equality Max: {eq_max:.4f}, Inequality Mean: {ieq_mean:.4f}, Inequality Max: {ieq_max:.4f}")

    return eq_mean, eq_max, ieq_mean, ieq_max


def summarize_feasibility(demand: torch.Tensor,
                              tech_feat: torch.Tensor,
                              flow: torch.Tensor,
                              pred_prod: torch.Tensor,
                              pred_flow: torch.Tensor,
                              tol: float = 1e-4,
                              print_summary: bool = False):
    """
    A Feasibility checker for single‐timestep ED:

    Output: Whether the 5 constraint is feasible or not.
    """
    
    if tech_feat.shape[1] != 4: 
        print(f"Check is only valid with investment info tech_feat shape {tech_feat.shape}")
        return

    # Feature are [Var_cost, Unit_Cap, Investmnet, Availability]
    investment = tech_feat[:, 2]
    availability = tech_feat[:, 3]
    unit_capacity = tech_feat[:, 1]
    p_max = availability * investment * unit_capacity         # if not a generator, p_max = 0
    total_demand = demand.sum()

    # 2) Check p bounds
    pred_prod = pred_prod.squeeze(-1)
    below_min_idx = (pred_prod < 0).nonzero(as_tuple=False).squeeze(-1)  # < 0
    above_max_idx = (pred_prod > p_max).nonzero(as_tuple=False).squeeze(-1)

    # 3) Check sum(p) = total_demand
    sum_p = pred_prod.sum()
    balance_violation = (sum_p - total_demand) < 0

    # 4) Check edge flows ∈ [0, capacity]
    # TODO: NEED TO CHANGE AFTER changing fllow representation
    import_cap = flow[:, 0]  # export capacity
    export_cap = flow[:, 1]  # import capacity
    pred_flow = pred_flow.squeeze(-1)  # Assuming pred_flow is already in the right shape
    pos_mask = pred_flow > 0
    neg_mask = pred_flow < 0
    #print(f"pred_flow {pred_flow} import cap {import_cap} export cap {export_cap} ")
    above_export_cap = (pred_flow > export_cap) #Inddex where flow exceeds export capacity
    below_import_cap = (pred_flow < import_cap) # Index whenre flow exceeds import capacity
    #print(f"above_export_cap {above_export_cap} below_import_cap {below_import_cap}")
    
    above_export_indices = (above_export_cap.nonzero(as_tuple=False).squeeze(-1)).tolist()
    below_import_indices = (below_import_cap.nonzero(as_tuple=False).squeeze(-1)).tolist()



    # ─── Print a short summary ───
    if print_summary:
        print("── Feasibility Check (no‐ramp) ──")
        if len(below_min_idx) or len(above_max_idx):
            if len(below_min_idx):
                print(f"  • p_i < 0 at nodes: {below_min_idx.tolist()}")
            if len(above_max_idx):
                print(f"  • p_i > p_max at nodes: {above_max_idx.tolist()}")
        else:
            print("  ✓ All p_i ∈ [0, p_max].")

        if balance_violation:
            print(f"  • Sum(p) = {sum_p:.4f}, but total_demand = {total_demand:.4f}")
        else:
            print("  ✓ Sum(p) = total_demand.")

        if len(above_export_indices) or len(below_import_indices):
            print("  • Flow violations:")
            if len(above_export_indices):
                for idx in above_export_indices:
                    print(f"    - Flow[{idx}] = {pred_flow[idx].item():.4f} > Export cap = {export_cap[idx].item():.4f}")
            if len(below_import_indices):
                for idx in below_import_indices:
                    print(f"    - Flow[{idx}] = {pred_flow[idx].item():.4f} < -Import cap = {-import_cap[idx].item():.4f}")
        else:
            print("  ✓ All flows ∈ [−import_cap, export_cap].")

    
    return {
        "p_below_zero": len(below_min_idx) == 0,
        "p_above_max":  len(above_max_idx) == 0,
        "sum_balance_violation": bool(balance_violation),
        "exceed_export_capacity": len(above_export_indices) == 0,
        "exceed_import_capacity":  len(below_import_indices) == 0,
    }
