import pandas as pd
import os

component_names = [ "cross_attn", "self_attn", "mix_ffn"]
results_folder = "/home/galkesten/transformers_project/ablation_results"
ablation_types = ["zero", "mean_over_tokens", "mean_per_token"]

for component_name in component_names:
    #connact all ablation types results into one dataframe
    all_results = pd.DataFrame()
    for ablation_type in ablation_types:
        results_file = f"{results_folder}/all_layers_results_{ablation_type}_{component_name}.csv"
        results = pd.read_csv(results_file)
        all_results = pd.concat([all_results, results])
    
    baseline_results = pd.read_csv(f"{results_folder}/baseline_results.csv")
    baseline_results["layer"] = "-"
    baseline_results["ablation_type"] = "baseline"
    baseline_results["ablation_component"] = "-"
    baseline_results.drop(columns=["experiment_type"], inplace=True)
    #print(baseline_results)
    #all_results = pd.concat([all_results, baseline_results])
    #print(all_results)
    layer_ranges = [0, 4, 9, 14, 19]  # upper bounds
    layer_labels = ["0-4", "5-9", "10-14", "15-19"]

    # Ensure 'layer' is numeric for comparison
    all_results["layer"] = pd.to_numeric(all_results["layer"], errors="coerce")

    # Map each layer number to a bucket
    all_results["layer_range"] = pd.cut(
        all_results["layer"],
        bins=[-1, 4, 9, 14, 19],   # -1 so that 0 falls in first bin
        labels=layer_labels)

    print(all_results)
    #create new col for layer label, for each row check which layer range it belongs to and add the label to the new col


    pivot_table_task_position_score = all_results.pivot_table(
        index=["ablation_type"],
        columns="layer",
        values="task_position_score"
        
    )

    pivot_table_overall_score = all_results.pivot_table(
        index=["ablation_type"],
        columns="layer",
        values="overall_score"
        
    )

    #create aggregated results_dir
    aggregated_results_dir = f"{results_folder}/aggregated_results"
    if not os.path.exists(aggregated_results_dir):
        os.makedirs(aggregated_results_dir)
    # save the pivot table to a csv file
    pivot_table_task_position_score.to_csv(f"{aggregated_results_dir}/all_layers_results_{component_name}_task_position_score.csv", index=True)
    pivot_table_overall_score.to_csv(f"{aggregated_results_dir}/all_layers_results_{component_name}_overall_score.csv", index=True)
    
    
    print(all_results)
    #group by layer_range and calculate the mean of the task_position_score and overall_score
    grouped_results = all_results.groupby(["ablation_type","ablation_component", "layer_range"]).mean()

    pivot_table_task_position_score_grouped = grouped_results.pivot_table(
        index=["ablation_type"],
        columns="layer_range",
        values="task_position_score"
        
    ) 

    pivot_table_overall_score_grouped = grouped_results.pivot_table(
        index=["ablation_type"],
        columns="layer_range",
        values="overall_score"
        
    )

    #save the pivot tables to csv files

    pivot_table_task_position_score_grouped.to_csv(f"{aggregated_results_dir}/all_layers_results_{component_name}_task_position_score_grouped.csv", index=True)
    pivot_table_overall_score_grouped.to_csv(f"{aggregated_results_dir}/all_layers_results_{component_name}_overall_score_grouped.csv", index=True)


for component_name in component_names:
    #connact all ablation types results into one dataframe
    all_results = pd.DataFrame()
    for ablation_type in ablation_types:
        results_file = f"{results_folder}/all_timesteps_results_{ablation_type}_{component_name}.csv"
        results = pd.read_csv(results_file)
        all_results = pd.concat([all_results, results])
    
    timestep_ranges = [0, 4, 9, 14, 19]  # upper bounds
    timestep_labels = ["0-4", "5-9", "10-14", "15-19"]

    # Ensure 'timestep is numeric for comparison
    all_results["timestep"] = pd.to_numeric(all_results["timestep"], errors="coerce")

    # Map each layer number to a bucket
    all_results["timestep_range"] = pd.cut(
        all_results["timestep"],
        bins=[-1, 4, 9, 14, 19],   # -1 so that 0 falls in first bin
        labels=layer_labels)

    print(all_results)
    #create new col for layer label, for each row check which layer range it belongs to and add the label to the new col


    pivot_table_task_position_score = all_results.pivot_table(
        index=["ablation_type"],
        columns="timestep",
        values="task_position_score"
        
    )

    pivot_table_overall_score = all_results.pivot_table(
        index=["ablation_type"],
        columns="timestep",
        values="overall_score"
        
    )

    #create aggregated results_dir
    aggregated_results_dir = f"{results_folder}/aggregated_results"
    if not os.path.exists(aggregated_results_dir):
        os.makedirs(aggregated_results_dir)
    # save the pivot table to a csv file
    pivot_table_task_position_score.to_csv(f"{aggregated_results_dir}/all_timesteps_results_{component_name}_task_position_score.csv", index=True)
    pivot_table_overall_score.to_csv(f"{aggregated_results_dir}/all_timesteps_results_{component_name}_overall_score.csv", index=True)
    
    
    print(all_results)
    #group by layer_range and calculate the mean of the task_position_score and overall_score
    grouped_results = all_results.groupby(["ablation_type","ablation_component", "timestep_range"]).mean()

    pivot_table_task_position_score_grouped = grouped_results.pivot_table(
        index=["ablation_type"],
        columns="timestep_range",
        values="task_position_score"
        
    ) 

    pivot_table_overall_score_grouped = grouped_results.pivot_table(
        index=["ablation_type"],
        columns="timestep_range",
        values="overall_score"
        
    )

    #save the pivot tables to csv files

    pivot_table_task_position_score_grouped.to_csv(f"{aggregated_results_dir}/all_timesteps_results_{component_name}_task_position_score_grouped.csv", index=True)
    pivot_table_overall_score_grouped.to_csv(f"{aggregated_results_dir}/all_timesteps_results_{component_name}_overall_score_grouped.csv", index=True)


