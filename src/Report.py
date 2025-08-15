import pandas as pd
import os
import numpy as np

class Report:
    def __init__(self, loss_cost=1000.0):
        self.rows = []  # per-instance summaries
        self.node_rows = []  # per-node outputs
        self.training_time = None
        self.inference_time = None
        self.loss_cost = loss_cost

    def add_instance(self, variable_cost, loss_of_load, instance_idx, production, production_gt,
                     flow, flow_gt, p_loss_gt, demand_feat, feasible):
        
        prod = production.squeeze()
        prod_gt = production_gt.squeeze()
    
        op_cost = np.dot(variable_cost, production)
        total = (variable_cost * production).sum()

        loss_cost_total = self.loss_cost * np.sum(loss_of_load[loss_of_load > 0])
        # print(f"Loss CostL {self.loss_cost} loss_of_load: {loss_of_load} , loss_cost_total: {loss_cost_total}")
        opt_loss_cost_total = self.loss_cost * np.sum(p_loss_gt[p_loss_gt > 0])

        total_obj = op_cost + loss_cost_total

        ratios = loss_of_load / (demand_feat)

        mean = ratios.mean()
        std = ratios.std()
        optimal_variable_cost = np.dot(variable_cost, production_gt)
        optimal_objective_value = self.loss_cost * np.sum(p_loss_gt[p_loss_gt >0]) + np.dot(variable_cost, production_gt)
        
        # Store summary row
        self.rows.append({
            "instance": instance_idx,
            "operational_cost": op_cost,
            "optimal_var_cost": optimal_variable_cost,
            "loss_cost": loss_cost_total,
            "optimal_loss_cost": opt_loss_cost_total,
            "objective_value": total_obj,
            "optimal_objective_value": optimal_objective_value,
            "feasible": feasible,
            "mean_loss_load_over_demand": mean,
            "std_loss_load_over_demand": std,
        })

        # Per-node production
        for i in range(len(prod)):
            self.node_rows.append({
                "instance": instance_idx,
                "node_type": "technology",
                "node_index": i,
                "prediction": prod[i].item(),
                "ground_truth": prod_gt[i].item()
            })

        # Per-node flow
        for i in range(len(flow)):
            self.node_rows.append({
                "instance": instance_idx,
                "node_type": "Flow",
                "node_index": i,
                "prediction": flow[i].item(),
                "ground_truth": flow_gt[i].item()
            })

    def make_report(self,
            save_path="./",
            filename="gnn_test_report.csv",
            node_filename="gnn_node_predictions.csv",
            scalars=None):
        import os
        import pandas as pd
        import numpy as np

        os.makedirs(save_path, exist_ok=True)

        # 1) Save the raw CSVs
        df_summary = pd.DataFrame(self.rows)
        df_summary["training_time"]  = self.training_time
        df_summary["inference_time"] = self.inference_time
        df_summary.to_csv(os.path.join(save_path, filename), index=False)

        df_nodes = pd.DataFrame(self.node_rows)
        df_nodes.to_csv(os.path.join(save_path, node_filename), index=False)

        print(f"Report saved to {os.path.join(save_path, filename)}")
        print(f"Node predictions saved to {os.path.join(save_path, node_filename)}")

        # 2) Unwrap any list or numpy-array columns of length 1 → scalar
        for col in df_summary.columns:
            if df_summary[col].apply(lambda x: isinstance(x, (list, np.ndarray))).all():
                df_summary[col] = df_summary[col].apply(
                    lambda x: x[0] if (isinstance(x, (list, np.ndarray)) and len(x) == 1) else x
                )

        # 3) Coerce numeric where possible
        numeric_df = df_summary.apply(pd.to_numeric, errors="ignore")

        # 4) Count & report feasibility rate
        total_count    = len(numeric_df)
        feas_df        = numeric_df[numeric_df["feasible"] == True]
        feasible_count = len(feas_df)
        pct_feasible   = (feasible_count / total_count * 100) if total_count else 0.0
        print(f"\nFeasible instances: {feasible_count}/{total_count} ({pct_feasible:.2f}%)")

        # 5) Column means (over feasible only)
        print("\nColumn Means for feasible==True (rounded to 4 decimals):")
        means = feas_df.mean(numeric_only=True).round(4)
        print(means)

        # 6) Compute per‐instance relative gaps (feasible only)
        #    gap_i = (objective_value_i - optimal_objective_value_i) / optimal_objective_value_i
        if feasible_count and "optimal_objective_value" in feas_df.columns:
            feas_df["rel_gap"] = (
                feas_df["objective_value"] - feas_df["optimal_objective_value"]
            ) / feas_df["optimal_objective_value"]
            avg_rel_gap = feas_df["rel_gap"].mean()*100
        else:
            avg_rel_gap = float("nan")

        # 7) Print scalars, totals, and average gap
        print(f"\nScalars: {scalars}")
        print(f"Average Relative Objective Gap (feasible only): {avg_rel_gap:.4f}%")
        print(f"Training time: {self.training_time:.4f} seconds")
