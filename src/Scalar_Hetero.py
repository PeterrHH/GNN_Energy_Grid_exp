# from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
# import torch
# import numpy as np

# class HeteroGraphScalar:
#     def __init__(self, scale = True):
#         '''
#         Scaler options thought of so far:
#         - MinMaxScaler
#         - StandardScaler(with_mean= False)
#         - MaxAbsScaler()
#         - RobustScaler()
#         '''
#         self.TechFeatScalar = MaxAbsScaler()
#         self.LocFeatScalar = MaxAbsScaler()
#         self.DemandFeatScalar = MaxAbsScaler()

#         # Separate scalers for flow features due to their distinct positive/negative nature
#         self.FlowNegFeatScalar = MaxAbsScaler() # For the negative flow capacity feature
#         self.FlowPosFeatScalar = MaxAbsScaler() # For the positive flow capacity feature

#         self.TechGtScalar = MaxAbsScaler()
#         self.FlowGtScalar = StandardScaler() 
#         self.scale = scale

#     def normalize_node_features(self, tech_features, loc_features, demand_features, flow_features):
#         """
#         Scale node features using appropriate scalers.
#         - Tech features: MaxAbsScaler, excluding the last column (availability).
#         - Location features: MaxAbsScaler.
#         - Demand features: MaxAbsScaler.
#         - Flow features: Separate MaxAbsScaler for negative and positive capacity components.
#         """
#         # Convert tensors to numpy for sklearn

#         # Scale tech features
#         if self.scale:
#             T, N, F = tech_features.shape
#             scale_features = tech_features[:, :, :-1].reshape(-1, F - 1)  # shape (T*N, F-1)
#             availability = tech_features[:, :, -1].reshape(T * N, 1)  # shape (T*N, 1)

#             scaled = self.TechFeatScalar.fit_transform(scale_features)
#             scaled_tensor = torch.tensor(scaled, dtype=torch.float32)

#             full_scaled = torch.cat([scaled_tensor, availability], dim=1)  # shape (T*N, F)
#             tech_features = full_scaled.view(T, N, F)

#             # Scale location features
#             loc_scaled = self.LocFeatScalar.fit_transform(loc_features.numpy())
#             loc_features = torch.tensor(loc_scaled, dtype=torch.float32)

#             # Scale demand features (shape: T x L)
#             # Assuming demand_features might be (T, L) or (T, L, 1) or similar.
#             # Reshape to (T*L, F_demand) if it has more than one feature per location
#             # If demand is (T, L), it should be reshaped to (T*L, 1) for scaler
#             T_dem, L_dem = demand_features.shape # Assuming (T, L)
#             demand_features_flat = demand_features.reshape(-1, 1).numpy()
#             demand_scaled = self.DemandFeatScalar.fit_transform(demand_features_flat)
#             demand_features = torch.tensor(demand_scaled, dtype=torch.float32).view(T_dem, L_dem)

#             # Scale flow features
#             # Assuming flow_features is (T, num_flow_nodes, 2)
#             N_flow, F_flow = flow_features.shape
#             flow_features_flat_np = flow_features.view(-1, F_flow).numpy()

#             # Split into negative and positive capacity features
#             flow_neg_capacity = flow_features_flat_np[:, 0].reshape(-1, 1)
#             flow_pos_capacity = flow_features_flat_np[:, 1].reshape(-1, 1)

#             # Scale independently
#             scaled_neg = self.FlowNegFeatScalar.fit_transform(flow_neg_capacity)
#             scaled_pos = self.FlowPosFeatScalar.fit_transform(flow_pos_capacity)

#             # Concatenate back
#             flow_scaled_combined = np.concatenate([scaled_neg, scaled_pos], axis=1)
#             flow_features = torch.tensor(flow_scaled_combined, dtype=torch.float32).view(N_flow, F_flow)

#         return tech_features, loc_features, demand_features, flow_features
        
#     def normalize_node_gt(self, tech_gt, flow_gt):
#         '''
#         Normalize ground truth values for Tech and Flow nodes over time.

#         tech_gt: shape (T, num_tech_nodes, 1)
#         flow_gt: shape (T, num_flow_nodes, 1)
#         '''
#         if self.scale:
#             T, N_tech, _ = tech_gt.shape
#             T, N_flow, _ = flow_gt.shape

#             # Flatten over time for fitting scaler
#             tech_np = tech_gt.view(-1, 1).numpy()
#             flow_np = flow_gt.view(-1, 1).numpy()

#             tech_scaled = self.TechGtScalar.fit_transform(tech_np)
#             flow_scaled = self.FlowGtScalar.fit_transform(flow_np) # Using StandardScaler here

#             tech_gt = torch.tensor(tech_scaled, dtype=torch.float32).view(T, N_tech, 1)
#             flow_gt = torch.tensor(flow_scaled, dtype=torch.float32).view(T, N_flow, 1)

#         return tech_gt, flow_gt

#     def inverse_transform(self, tech_pred, flow_pred):
#         '''
#         Inverse transform predicted Tech and Flow node outputs.

#         tech_pred: shape (T, num_tech_nodes, 1)
#         flow_pred: shape (T, num_flow_nodes, 1)
#         '''
#         if self.scale:
#             T_tech, N_tech = tech_pred.shape[:2] # Can be (T, N_tech) or (T, N_tech, 1)
#             T_flow, N_flow = flow_pred.shape[:2] # Can be (T, N_flow) or (T, N_flow, 1)
            
#             tech_np = tech_pred.view(-1, 1).detach().numpy()
#             flow_np = flow_pred.view(-1, 1).detach().numpy()

#             tech_inv = self.TechGtScalar.inverse_transform(tech_np)
#             flow_inv = self.FlowGtScalar.inverse_transform(flow_np) # Using StandardScaler here

#             tech_pred = torch.tensor(tech_inv, dtype=torch.float32).view(T_tech, N_tech, -1)
#             flow_pred = torch.tensor(flow_inv, dtype=torch.float32).view(T_flow, N_flow, -1)

#         return tech_pred, flow_pred
        
#     def inverse_data(self, data):
#         """
#         Inverse transforms features for a single (non-batched) HeteroData graph.
#         Assumes features are already flattened across time if multiple timesteps were processed
#         before saving to `data` object.
#         """
#         tech_features = data['technology'].x # Assumed shape (N_total_tech, F_tech)
#         loc_features = data['location'].x # Assumed shape (N_loc, F_loc)
#         demand_features = data['demand'].x # Assumed shape (N_total_demand, F_demand)
#         flow_features = data['flow'].x # Assumed shape (N_total_flow, F_flow=2)
#         if self.scale:
#             # Inverse transform tech features
#             N_tech_total, F_tech = tech_features.shape
#             scale_features_np = tech_features[:, :-1].detach().numpy()  # (N_total_tech, F-1)
#             availability = tech_features[:, -1].unsqueeze(-1)           # (N_total_tech, 1)

#             tech_inv_scaled = self.TechFeatScalar.inverse_transform(scale_features_np)  # (N_total_tech, F-1)
#             tech_full = np.concatenate([tech_inv_scaled, availability.numpy()], axis=1) # (N_total_tech, F)
#             tech_features = torch.tensor(tech_full, dtype=torch.float32)


#             # Inverse transform demand features
#             # Assuming demand_features is (N_total_demand, 1)
#             demand_np = demand_features.detach().numpy()
#             demand_inv = self.DemandFeatScalar.inverse_transform(demand_np)
#             demand_features = torch.tensor(demand_inv, dtype=torch.float32)

#             # Inverse transform flow features
#             # Assuming flow_features is (N_total_flow, 2)
#             flow_np = flow_features.detach().numpy()
#             flow_neg_scaled = flow_np[:, 0].reshape(-1, 1)
#             flow_pos_scaled = flow_np[:, 1].reshape(-1, 1)

#             flow_inv_neg = self.FlowNegFeatScalar.inverse_transform(flow_neg_scaled)
#             flow_inv_pos = self.FlowPosFeatScalar.inverse_transform(flow_pos_scaled)

#             flow_inv_combined = np.concatenate([flow_inv_neg, flow_inv_pos], axis=1)
#             flow_features = torch.tensor(flow_inv_combined, dtype=torch.float32)

#         return tech_features, demand_features, flow_features


#     def inverse_batch_data(self, batch, T):
#         """
#         Inverse transform all node features in a batched HeteroData object.
#         Assumes features in 'batch' are already flattened across time,
#         i.e., 'technology'.x is (T*N_tech, F_tech).

#         Assumes:
#         - 'technology' features last column is availability and not scaled.
#         - 'demand' and 'flow' are fully scaled.
#         """
        
#         # Inverse transform tech features
#         tech_features = batch['technology'].x
#         demand_features = batch['demand'].x
#         flow_features = batch['flow'].x
#         N_tech = tech_features.shape[0] // T # Calculate original N_tech per timestep
#         N_demand_nodes_per_timestep = demand_features.shape[0] // T
#         N_flow = flow_features.shape[0] // T

#         if self.scale:
#             tech_scaled = tech_features[:, :-1].detach().numpy()
#             tech_avail = tech_features[:, -1].unsqueeze(-1)  # keep as is
#             tech_inv = self.TechFeatScalar.inverse_transform(tech_scaled)
#             tech_full = np.concatenate([tech_inv, tech_avail.numpy()], axis=1)
#             tech_inverse = torch.tensor(tech_full, dtype=torch.float32)
#             tech_inverse = tech_inverse.view(T, N_tech, -1) # Reshape back to (T, N_tech, F)



#             # Inverse transform demand features (assume demand is shape (N_total_demand, 1))


#             demand_np = demand_features.detach().numpy()
#             demand_inv = self.DemandFeatScalar.inverse_transform(demand_np)
#             demand_inverse = torch.tensor(demand_inv, dtype=torch.float32).view(T, N_demand_nodes_per_timestep, -1).squeeze()
            
#             # Inverse transform flow features
    

#             flow_features = flow_features.view(T, N_flow, -1)[0]
#             flow_features_flat_np = flow_features.detach().numpy() # (T * N_flow, 2)

#             flow_neg_scaled = flow_features_flat_np[:, 0].reshape(-1, 1)
#             flow_pos_scaled = flow_features_flat_np[:, 1].reshape(-1, 1)

#             flow_inv_neg = self.FlowNegFeatScalar.inverse_transform(flow_neg_scaled)
#             flow_inv_pos = self.FlowPosFeatScalar.inverse_transform(flow_pos_scaled)

#             flow_inv_combined = np.concatenate([flow_inv_neg, flow_inv_pos], axis=1)
#             flow_inverse= torch.tensor(flow_inv_combined, dtype=torch.float32)

#             return tech_inverse, demand_inverse, flow_inverse
#         else:
#             tech_features = tech_features.view(T, N_tech, -1)
#             demand_features = demand_features.view(T, N_demand_nodes_per_timestep, -1)
#             flow_features = flow_features.view(T, N_flow, -1)
#             return tech_features, demand_features, flow_features
        


# '''

# -------------Feasibility Check Example------------------
# Tech features shape: torch.Size([5000, 5, 4]), Demand features shape: torch.Size([5000, 3]), Flow features shape: torch.Size([2, 2])
# '''
# per_unit_hetero_scalar.py
import torch
import numpy as np
from typing import Optional

class HeteroGraphScalar:
    """
    Single-base (per-unit) scaler for power-like quantities.
    Scales demand, production GT, flow GT, line capacities, and unit_capacity
    by one scalar P_base so balance holds in scaled space.

    Choose how to set P_base with base_strategy:
      - "percentile" (default, q=90.0)
      - "mean"
      - "rms"
      - "fixed" (provide fixed_base)

    Variable cost, investment, availability are left unchanged.
    Location features are untouched.
    """

    def __init__(
        self,
        scale: bool = True,
        base_strategy: str = "percentile",
        q: float = 50.0,
        fixed_base: Optional[float] = None,
        min_base: float = 1e-6
    ):
        self.scale = scale
        self.base_strategy = base_strategy
        self.q = q
        self.fixed_base = fixed_base
        self.min_base = min_base
        self.P_base: Optional[float] = None

    # ---------- internal helpers ----------
    def _fit_base(self, tech_features: torch.Tensor, demand_features: torch.Tensor, flow_features: torch.Tensor):
        if not self.scale or self.P_base is not None:
            return

        # demand_features: [T, N_loc]
        demand_np = demand_features.detach().cpu().numpy().astype(float).reshape(-1)
        # flow_features: [E, 2] (signed or magnitudes; we use magnitudes for scaling)
        flow_np = np.abs(flow_features.detach().cpu().numpy().astype(float).reshape(-1))
        # tech_features: [T, N_tech, 4], use a proxy for p_max scale (unit_cap * availability)
        unit_cap = tech_features[:, :, 1].detach().cpu().numpy().astype(float).reshape(-1)
        avail    = tech_features[:, :, 3].detach().cpu().numpy().astype(float).reshape(-1)
        pmax_proxy = unit_cap * avail

        # pool determines the scale statistics (mostly driven by demand)
        pool = np.concatenate([np.abs(demand_np), flow_np, np.abs(pmax_proxy)])

        if self.base_strategy == "fixed":
            base = float(self.fixed_base if self.fixed_base is not None else 1.0)
        elif self.base_strategy == "mean":
            base = float(np.mean(pool))
        elif self.base_strategy == "rms":
            base = float(np.sqrt(np.mean(pool ** 2)))
        else:  # "percentile"
            base = float(np.percentile(pool, self.q))

        # guard
        if not np.isfinite(base) or base <= 0:
            base = 1.0
        self.P_base = max(base, self.min_base)

    def _scale_power(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.P_base if (self.scale and self.P_base) else x

    def _inv_power(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.P_base if (self.scale and self.P_base) else x

    # ---------- API: features ----------
    def normalize_node_features(self, tech_features, loc_features, demand_features, flow_features):
        """
        tech_features:  [T, N_tech, 4] -> [var_cost, unit_capacity, investment, availability]
        loc_features:   [N_loc, F_loc] (unchanged)
        demand_features:[T, N_loc]
        flow_features:  [E, 2]  (directional caps; signs preserved here)
        """
        if not self.scale:
            return tech_features, loc_features, demand_features, flow_features

        self._fit_base(tech_features, demand_features, flow_features)

        tech_scaled   = tech_features.clone().to(torch.float32)
        loc_scaled    = loc_features.to(torch.float32)
        demand_scaled = self._scale_power(demand_features.to(torch.float32))
        flow_scaled   = self._scale_power(flow_features.to(torch.float32))  # both columns by same base

        # scale ONLY unit_capacity (index 1). cost/invest/availability unchanged.
        tech_scaled[:, :, 1] = self._scale_power(tech_scaled[:, :, 1])

        return tech_scaled, loc_scaled, demand_scaled, flow_scaled

    # ---------- API: ground truths ----------
    def normalize_node_gt(self, tech_gt, flow_gt):
        """
        tech_gt: [T, N_tech, 1]  (production)
        flow_gt: [T, N_flow, 1]
        """
        if not self.scale:
            return tech_gt.to(torch.float32), flow_gt.to(torch.float32)
        print(f"Flow gt {flow_gt.shape}")
        tech_gt = self._scale_power(tech_gt.to(torch.float32))
        flow_gt = self._scale_power(flow_gt.to(torch.float32))

        return tech_gt, flow_gt

    # ---------- API: inverse predictions ----------
    def inverse_transform(self, tech_pred, flow_pred):
        """
        Inverse transform predicted Tech and Flow node outputs.
        Accepts shapes [T, N, 1] or [T, N].
        """
        if not self.scale:
            return tech_pred, flow_pred
        return self._inv_power(tech_pred.to(torch.float32)), self._inv_power(flow_pred.to(torch.float32))

    # ---------- API: inverse features for a single graph ----------
    def inverse_data(self, data):
        """
        Returns (tech_features, demand_features, flow_features) in physical units (MW).
        Expects:
          data['technology'].x: [N_tech, 4]
          data['demand'].x:     [N_loc, 1] or [N_loc]
          data['flow'].x:       [E, 2]
        """
        tech_x   = data['technology'].x.to(torch.float32).clone()
        demand_x = data['demand'].x.to(torch.float32).clone()
        flow_x   = data['flow'].x.to(torch.float32).clone()

        if not self.scale:
            return tech_x, demand_x.squeeze(-1), flow_x

        tech_x[:, 1] = self._inv_power(tech_x[:, 1])     # unit_capacity back to MW
        demand_inv   = self._inv_power(demand_x).squeeze(-1)
        flow_inv     = self._inv_power(flow_x)
        return tech_x, demand_inv, flow_inv

    # ---------- API: inverse features for a batched graph ----------
    def inverse_batch_data(self, batch, T):
        """
        Returns:
          tech_inverse:   [T, N_tech, 4]
          demand_inverse: [T, N_loc]
          flow_inverse:   [T, N_flow, 2]
        """
        tech_x   = batch['technology'].x.to(torch.float32).clone()
        demand_x = batch['demand'].x.to(torch.float32).clone()
        flow_x   = batch['flow'].x.to(torch.float32).clone()

        N_tech = tech_x.shape[0] // T
        N_loc  = demand_x.shape[0] // T
        N_flow = flow_x.shape[0] // T

        tech_x   = tech_x.view(T, N_tech, -1)
        demand_x = demand_x.view(T, N_loc, -1)
        flow_x   = flow_x.view(T, N_flow, -1)

        if not self.scale:
            return tech_x, demand_x.squeeze(-1), flow_x

        tech_inv = tech_x.clone()
        tech_inv[:, :, 1] = self._inv_power(tech_inv[:, :, 1])

        demand_inv = self._inv_power(demand_x).squeeze(-1)
        flow_inv   = self._inv_power(flow_x)
        return tech_inv, demand_inv, flow_inv
