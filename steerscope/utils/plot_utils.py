import os, glob, random, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import auc
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import pandas as pd
from pathlib import Path                        
import numpy as np
import argparse
import pickle
from plotnine import *
import json
import pandas as pd
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
import itertools
import numpy as np
from scipy.stats import bootstrap, ttest_rel
from collections import defaultdict, Counter
import glob

from tqdm import tqdm
import random


def plot_metrics(jsonl_data, configs, write_to_path=None, report_to=[], wandb_name=None, mode=None):
    # Collect data into a list
    data = []
    for config in configs:
        evaluator_name = config['evaluator_name']
        metric_name = config['metric_name']
        y_label = config['y_label']
        use_log_scale = config['use_log_scale']
        
        for entry in jsonl_data:
            results = entry.get('results', {}).get(evaluator_name, {})
            for method, res in results.items():
                factors = res.get('factor', [])
                metrics = res.get(metric_name, [])
                # Ensure factors and metrics are lists
                if not isinstance(factors, list):
                    factors = [factors]
                if not isinstance(metrics, list):
                    metrics = [metrics]
                for f, m in zip(factors, metrics):
                    data.append({
                        'Factor': f,
                        'Value': m,
                        'Method': method,
                        'Metric': y_label,
                        'UseLogScale': use_log_scale
                    })

    # Create DataFrame and average metrics
    df = pd.DataFrame(data)
    df = df.groupby(['Method', 'Factor', 'Metric', 'UseLogScale'], as_index=False).mean()

    # Apply log transformation if needed
    df['TransformedValue'] = df.apply(
        lambda row: np.log10(row['Value']) if row['UseLogScale'] else row['Value'],
        axis=1
    )

    # Create the plot
    p = (
        ggplot(df, aes(x='Factor', y='TransformedValue', color='Method', group='Method')) +
        geom_line() +
        geom_point() +
        theme_bw() +
        labs(x='Factor', y='Value') +
        facet_wrap('~ Metric', scales='free_y', nrow=1) +  # Plots in a row
        theme(
            subplots_adjust={'wspace': 0.1},
            figure_size=(1.5 * len(configs), 3),  # Wider for more plots, taller height
            legend_position='right',
            legend_title=element_text(size=4),
            legend_text=element_text(size=6),
            axis_title=element_text(size=6),
            axis_text=element_text(size=6),
            axis_text_x=element_text(rotation=90, hjust=1),  # Rotate x-axis labels
            strip_text=element_text(size=6)
        )
    )

    # Save or show the plot
    if write_to_path:
        p.save(filename=str(write_to_path / f"{mode}_plot.png"), dpi=300, bbox_inches='tight')
    else:
        print(p)

    # Report to wandb if wandb_name is provided
    if report_to is not None and "wandb" in report_to:
        import wandb
        # Separate data by metrics to prepare for wandb line series plotting
        line_series_plots = {}
        for metric in df['Metric'].unique():
            metric_data = df[df['Metric'] == metric]
            
            xs = metric_data['Factor'].unique().tolist()
            ys = [metric_data[metric_data['Method'] == method]['TransformedValue'].tolist() for method in metric_data['Method'].unique()]
            keys = [f"{method}" for method in metric_data['Method'].unique()]
            
            line_series_plots[f"{mode}/{metric}"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title=f"{metric}",
                xname='Factor'
            )
        wandb.log(line_series_plots)

def plot_metrics_multiple_datasets(data_path, write_to_path=None, report_to=[], wandb_name=None, mode=None, rule=True):
    # Collect data into a list
    df = pd.read_parquet(data_path)

    for method in [a.split('_')[0] for a in list(df.columns) if "LMJudgeEvaluator_relevance_concept_ratings" in a]:
        col = method + '_LMJudgeEvaluator_relevance_concept_ratings'
        norm_col = method + '_normalized_LMJudgeEvaluator_relevance_concept_ratings'
        df[norm_col] = [0 for _ in range(len(df))]
        for dataset in df['dataset_name'].unique():
            mask_dataset = (df['dataset_name'] == dataset)
            for concept_id in df.loc[mask_dataset, 'concept_id'].unique():
                mask_concept = mask_dataset & (df['concept_id'] == concept_id)
                for input_id in df.loc[mask_concept, 'input_id'].unique():
                    mask_input = mask_concept & (df['input_id'] == input_id)
                    mask_minus2 = mask_input & (df['factor'] == -2)
                    if mask_minus2.any():
                        base_val = 2-df.loc[mask_minus2, col].values[0]
                        val = (2 - df.loc[mask_input, col]) - base_val
                        val = val.clip(lower=0)
                        df.loc[mask_input, norm_col] = val


    # Compute harmonic mean for normalized LMJudgeEvaluator metrics and store in new column
    for method in [a[:len(a)-len("_LMJudgeEvaluator_relevance_concept_ratings")] for a in list(df.columns) if "LMJudgeEvaluator_relevance_concept_ratings" in a]:
        norm_concept_col = method + '_normalized_LMJudgeEvaluator_relevance_concept_ratings'
        instr_col = method + '_LMJudgeEvaluator_relevance_instruction_ratings'
        fluency_col = method + '_LMJudgeEvaluator_fluency_ratings'
        new_col = method + '_normalized_LMJudgeEvaluator'
        def safe_hmean(row):
            vals = [row.get(norm_concept_col, 0), row.get(instr_col, 0), row.get(fluency_col, 0)]
            if 0 in vals:
                return 0
            hmean = 3 / sum(1/v for v in vals)
            return hmean if hmean >= 0 else 0
        df[new_col] = df.apply(safe_hmean, axis=1)

    def harmonic_mean(scores):
        # Return 0 if any score is 0 to maintain strict evaluation
        if 0 in scores:
            return 0
        return len(scores) / sum(1/s for s in scores)

    methods = [a[:len(a)-len("_normalized_LMJudgeEvaluator_relevance_concept_ratings")] for a in list(df.columns) if "LMJudgeEvaluator_relevance_concept_ratings" in a]


    # Metrics to plot
    if not rule:
        metrics = ['_normalized_LMJudgeEvaluator',
                '_normalized_LMJudgeEvaluator_relevance_concept_ratings', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Normalized Overall', 'Normalized Relevance Concept', 'Relevance Instruction', 'Fluency']
    
    else:
       
        metrics = [
                '_RuleEvaluator',
                '_RuleEvaluator_rule_following', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Rule Following', 'Relevance Instruction', 'Fluency']
    
    
    
    # Prepare data for plotting
    plot_data = []
    datasets = df['dataset_name'].unique()
    # For each dataset, calculate average metrics for each factor
    for dataset in datasets:
        dataset_data = df[df['dataset_name'] == dataset]     
        # Group by method and factor, then calculate mean for each metric
        for method in methods:
            if 'PromptSteering' in method:
                # For PromptSteering, average across all factors
                # Calculate average for each metric
                for idx, metric in enumerate(metrics):
                    if method + metric in dataset_data.columns:
                        avg_value = dataset_data[method + metric].mean()
                        # Add same average value for each factor to create a straight line
                        for factor in dataset_data['factor'].unique():
                            plot_data.append({
                                'Dataset': dataset,
                                'Method': method,
                                'Factor': factor,
                                'Metric': metrics_names[idx],
                                'Value': avg_value
                            })
            else:
                # For other methods, keep factor-wise values
                for factor in dataset_data['factor'].unique():
                    factor_data = dataset_data[dataset_data['factor'] == factor]
                    # Calculate average for each metric
                    for idx, metric in enumerate(metrics):
                        if method + metric in factor_data.columns:
                            avg_value = factor_data[method + metric].mean()
                            plot_data.append({
                                'Dataset': dataset,
                                'Method': method,
                                'Factor': factor,
                                'Metric': metrics_names[idx],
                                'Value': avg_value
                            })
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Ensure metrics are displayed in the correct order by creating a categorical variable
    plot_df['Metric'] = pd.Categorical(plot_df['Metric'], categories=metrics_names, ordered=True)
    
    # Create a plot for each dataset
    num_datasets = len(datasets)
    

    # Get unique combinations of Dataset and Metric
    unique_combinations = plot_df[['Dataset', 'Metric']].drop_duplicates()
    num_combinations = len(unique_combinations)

    # Calculate grid dimensions
    ncols = 4
    nrows = int(np.ceil(num_combinations / ncols))

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    # Get unique methods for color mapping
    unique_methods = plot_df['Method'].unique()
    color_positions = (
        np.linspace(0.03, 0.97, len(unique_methods))
        if len(unique_methods) > 1
        else [0.5]
    )
    colors = plt.get_cmap('turbo')(color_positions)
    method_colors = dict(zip(unique_methods, colors))
    markers = ('o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', '*', 'p')
    linestyles = ('-', '--', '-.', ':')
    method_styles = {
        method: (
            markers[index % len(markers)],
            linestyles[(index // len(markers)) % len(linestyles)],
        )
        for index, method in enumerate(unique_methods)
    }

    # Plot each combination
    for idx, (dataset, metric) in enumerate(unique_combinations.values):
        ax = axes[idx]
        subset = plot_df[(plot_df['Dataset'] == dataset) & (plot_df['Metric'] == metric)]
        
        # Plot each method
        for method in unique_methods:
            method_data = subset[subset['Method'] == method]
            if not method_data.empty:
                ax.plot(method_data['Factor'], method_data['Value'], 
                    color=method_colors[method], label=method,
                    marker=method_styles[method][0],
                    linestyle=method_styles[method][1])
        
        ax.set_title(f'{metric}')
        ax.set_xlabel('Factor')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # Add legend to the last subplot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5))

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if write_to_path:
        plt.savefig(str(write_to_path / f"{mode}_combined_plot.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    return fig 


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path

# Define a better color palette - using colorblind-friendly colors with more distinct blues
COLORS = {
    'PreferenceVector': '#404040',              # Dark gray
    'PromptSteering_prepend_original': '#00BFFF', # Deep sky blue
    'PromptSteering_prepend_rewrite': '#0000CD',  # Medium blue
    'PromptSteering_append_original': '#DA70D6',  # Orchid
    'PromptSteering_append_rewrite': '#F15C5C'    # Salmon red
}

def get_nshot_one_df(data_path, write_to_path=None, report_to=[], wandb_name=None, mode=None, rule=True):
    df = pd.read_parquet(data_path)   
    # Extract shot numbers from dataset names
    df['n_shots'] = df['input'].str.count('Question')  
    methods = [a[:len(a)-len("_LMJudgeEvaluator_relevance_concept_ratings")] for a in list(df.columns) if "LMJudgeEvaluator_relevance_concept_ratings" in a]

    # Metrics to plot
    if not rule:
        metrics = ['_LMJudgeEvaluator',
                '_LMJudgeEvaluator_relevance_concept_ratings', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Adherence to System Prompt', 'Relevance Instruction', 'Fluency']
    
    else:
        metrics = [
                '_RuleEvaluator',
                '_RuleEvaluator_rule_following', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Adherence to System Prompt', 'Relevance Instruction', 'Fluency']    
    
    # Prepare data for plotting
    plot_data = []
    
    # Calculate average metrics for each method and number of shots
    for method in methods:
        for n_shots in df['n_shots'].unique():
            shots_data = df[df['n_shots'] == n_shots]
            # Calculate average for each metric
            for idx, metric in enumerate(metrics):
                if method + metric in shots_data.columns:
                    avg_value = shots_data[method + metric].mean()
                    if "_RuleEvaluator_rule_following" in metric:
                        avg_value = 2 - avg_value
                    plot_data.append({
                        "Dataset": "AttackMultiShot",
                        'n_shots': n_shots,
                        'Method': method,
                        'Metric': metrics_names[idx],
                        'Value': avg_value
                    })
    return plot_data
    
def plot_nshot_metrics(data_paths, write_to_path=None, report_to=[], wandb_name=None, mode=None, rule=True):
    """
    Create an n-shot metrics plot with confidence intervals, similar to the style in analyse.ipynb.
    Only shows the "Overall" metric.
    """
    plot_data = []
    for data_path in data_paths:
        plot_data.extend(get_nshot_one_df(data_path, write_to_path, report_to, wandb_name, mode, rule))
 
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by number of shots to ensure proper ordering on x-axis
    plot_df = plot_df.sort_values('n_shots')
    
    # Debug: print unique n_shots values
    print(f"Unique n_shots values: {sorted(plot_df['n_shots'].unique())}")
    print(f"Data shape: {plot_df.shape}")
    print(f"Value range: {plot_df['Value'].min()} to {plot_df['Value'].max()}")
    
    # Convert n_shots to string for discrete treatment if there are few unique values
    unique_shots = sorted(plot_df['n_shots'].unique())
    if len(unique_shots) <= 10:  # Treat as discrete if 10 or fewer unique values
        plot_df['n_shots_str'] = plot_df['n_shots'].astype(str)
        x_var = 'n_shots_str'
    else:
        x_var = 'n_shots'
    
    # Get unique methods for color mapping
    unique_methods = sorted(plot_df['Method'].unique())
    
    # Define the preferred order of methods (for legend)
    method_order = [
        'PromptSteering_prepend_original',
        'PromptSteering_prepend_rewrite',
        'PromptSteering_append_original',
        'PromptSteering_append_rewrite',
        'PreferenceVector'
    ]
    
    # Filter to only include methods that are actually in the data
    method_order = [m for m in method_order if m in unique_methods]
    
    # Create method labels for better readability
    method_labels = {method: method.replace('_', ' ') for method in unique_methods}
    plot_df['method_label'] = plot_df['Method'].map(method_labels)
    
    # Keep only the 'Overall' metric for plotting
    plot_df = plot_df[plot_df['Metric'] == 'Overall']
    
    # Create the color scale using the global COLORS dictionary
    method_colors = {method: COLORS.get(method, "#333333") for method in unique_methods}
    
    # Create the plot using plotnine - similar to analyse.ipynb style
    plot = (
        ggplot(plot_df, aes(x=x_var, y='Value', color='Method', group='Method')) +
        stat_summary(fun_data="mean_cl_boot", geom="errorbar", width=0.2, alpha=0.7) +
        stat_summary(fun_data="mean_cl_boot", geom="line") +
        stat_summary(fun_data="mean_cl_boot", geom="point") +
        labs(
            x='Number of Shots',
            y='Overall Score',
            title='Effect of Number of Shots on Overall Performance'
        ) +
        scale_color_manual(values=method_colors)
    )
    
    # If using discrete x-axis, adjust the plot
    if x_var == 'n_shots_str':
        plot = plot + scale_x_discrete()
    else:
        # Check if log scale might be appropriate (if values span more than one order of magnitude)
        min_val = min(unique_shots)
        max_val = max(unique_shots)
        plot = plot + scale_x_continuous(
            breaks=sorted(plot_df['n_shots'].unique()),
            limits=(plot_df['n_shots'].min() - 0.5, plot_df['n_shots'].max() + 0.5)
        )
    
    # Apply theme similar to analyse.ipynb but with simpler legend settings
    plot = plot + theme_bw() + theme(
        text=element_text(family="sans-serif"),
        plot_title=element_text(size=14, hjust=0.5),
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
        legend_position="bottom",
        legend_title=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(fill=None, color="black", size=0.5),
        figure_size=(8, 6)
    )
    
    # Save or show the plot
    if write_to_path:
        output_path = Path(write_to_path) / f"{mode}_n_shot_plot.pdf"
        plot.save(str(output_path), dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        print(plot)
    
    return plot


def plot_win_rates(jsonl_data, write_to_path=None, report_to=[], wandb_name=None):
    # Collect methods and baseline models
    methods = set()
    baseline_models = set()
    for entry in jsonl_data:
        winrate_results = entry.get('results', {}).get('WinRateEvaluator', {})
        for method_name, res in winrate_results.items():
            methods.add(method_name)
            baseline_models.add(res.get('baseline_model', 'Unknown'))
    methods = sorted(list(methods))
    baseline_models = sorted(list(baseline_models))
    
    # Assuming all methods are compared against the same baseline
    if len(baseline_models) == 1:
        baseline_model = baseline_models[0]
    else:
        # Handle multiple baselines if necessary
        baseline_model = baseline_models[0]
    
    # Add the baseline method to methods if not already present
    if baseline_model not in methods:
        methods.append(baseline_model)
    
    # Initialize data structures
    win_rates = {method: [] for method in methods}
    loss_rates = {method: [] for method in methods}
    tie_rates = {method: [] for method in methods}
    
    # Collect data from all concepts
    num_concepts = len(jsonl_data)
    for entry in jsonl_data:
        winrate_results = entry.get('results', {}).get('WinRateEvaluator', {})
        for method in methods:
            if method == baseline_model:
                continue  # Handle baseline separately
            if method in winrate_results:
                res = winrate_results[method]
                win_rates[method].append(res.get('win_rate', 0) * 100)
                loss_rates[method].append(res.get('loss_rate', 0) * 100)
                tie_rates[method].append(res.get('tie_rate', 0) * 100)
            else:
                # If method is not present in this concept, assume zero rates
                win_rates[method].append(0.0)
                loss_rates[method].append(0.0)
                tie_rates[method].append(0.0)
    
    # For the baseline method, set win_rate=50%, loss_rate=50%, tie_rate=0%
    win_rates[baseline_model] = [50.0] * num_concepts
    loss_rates[baseline_model] = [50.0] * num_concepts
    tie_rates[baseline_model] = [0.0] * num_concepts
    
    # Calculate mean percentages
    win_means = {method: np.mean(vals) for method, vals in win_rates.items()}
    loss_means = {method: np.mean(vals) for method, vals in loss_rates.items()}
    tie_means = {method: np.mean(vals) for method, vals in tie_rates.items()}
    
    # Sort methods: baseline at top, then methods by descending win rate
    non_baseline_methods = [m for m in methods if m != baseline_model]
    sorted_methods = sorted(
        non_baseline_methods,
        key=lambda m: win_means[m],
        reverse=True
    )
    
    # Prepare data for plotting
    data = []
    for method in sorted_methods:
        data.append({'Method': method, 'Outcome': 'Loss', 'Percentage': loss_means[method]})
        data.append({'Method': method, 'Outcome': 'Tie', 'Percentage': tie_means[method]})
        data.append({'Method': method, 'Outcome': 'Win', 'Percentage': win_means[method]})
    
    df = pd.DataFrame(data)
    
    # Set the order of Outcome to control stacking order
    df['Outcome'] = pd.Categorical(df['Outcome'], categories=['Loss', 'Tie', 'Win'], ordered=True)
    # Reverse the methods list for coord_flip to display baseline at the top
    df['Method'] = pd.Categorical(df['Method'], categories=sorted_methods[::-1], ordered=True)
    
    # Ensure df is sorted properly
    df = df.sort_values(['Method', 'Outcome'])
    # Convert 'Percentage' to float
    df['Percentage'] = df['Percentage'].astype(float)
    
    # Compute cumulative percentage per method
    df['cum_percentage'] = df.groupby('Method')['Percentage'].cumsum()
    # Shift cumulative percentages per method
    df['cum_percentage_shifted'] = df.groupby('Method')['cum_percentage'].shift(1).fillna(0)
    
    # For the 'Win' outcome, get the cumulative percentage up to before 'Win'
    df_win = df[df['Outcome'] == 'Win'].copy()
    df_win['text_position'] = df_win['cum_percentage_shifted']
    # Convert 'text_position' to float
    df_win['text_position'] = 100.0 - df_win['text_position'].astype(float)
    # Format the win percentage label
    df_win['win_percentage_label'] = df_win['Percentage'].map(lambda x: f"{x:.1f}%")
    
    # Create the plot
    p = (
        ggplot(df, aes(x='Method', y='Percentage', fill='Outcome')) +
        geom_bar(stat='identity', position='stack', width=0.8) +
        # Add the geom_text layer to include win rate numbers
        geom_text(
            data=df_win,
            mapping=aes(
                x='Method',
                y='text_position',
                label='win_percentage_label'
            ),
            ha='right',
            va='center',
            size=6,  # Adjust size as needed
            color='black',
            nudge_y=18  # Adjust this value as needed for proper positioning
        ) +
        coord_flip() +  # Flip coordinates for horizontal bars
        theme_bw() +
        labs(
            y='Percentage (%)',
            x=''
        ) +
        theme(
            axis_text_x=element_text(size=6),
            axis_text_y=element_text(size=6),
            axis_title=element_text(size=6),
            legend_title=element_text(size=6),
            legend_text=element_text(size=6),
            figure_size=(3, len(sorted_methods) * 0.3 + 0.3)
        ) +
        scale_fill_manual(
            values={'Win': '#a6cee3', 'Tie': '#bdbdbd', 'Loss': '#fbb4ae'},
            guide='legend',
            name='Outcome'
        )
    )
    
    # Save or show the plot
    if write_to_path:
        p.save(filename=str(write_to_path / "winrate_plot.png"), dpi=300, bbox_inches='tight')
    else:
        print(p)
    
    if report_to is not None and "wandb" in report_to:
        import wandb
        wandb.log({"steering/winrate_plot": wandb.Image(str(write_to_path / "winrate_plot.png"))})


def plot_best_factor_scores(data_path, write_to_path=None, report_to=[], wandb_name=None, mode=None, rule=True):
    # Read the data
    df = pd.read_parquet(data_path)
    
    def harmonic_mean(scores):
        # Return 0 if any score is 0 to maintain strict evaluation
        if 0 in scores:
            return 0
        return len(scores) / sum(1/s for s in scores)
    
    methods = [a[:len(a)-len("_LMJudgeEvaluator_relevance_concept_ratings")] for a in list(df.columns) if "LMJudgeEvaluator_relevance_concept_ratings" in a]
    df = df[df['dataset_name'] == 'AlpacaEvalSuppress']

    # Metrics to plot
    if not rule:
        metrics = ['_LMJudgeEvaluator',
                '_LMJudgeEvaluator_relevance_concept_ratings', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Relevance Concept', 'Relevance Instruction', 'Fluency']
    else:
        # Calculate harmonic mean for each method and add as a new column
       
        metrics = [
                '_RuleEvaluator',
                '_RuleEvaluator_rule_following', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Rule Following', 'Relevance Instruction', 'Fluency']
    
    # Get all concepts
    concepts = df['concept_id'].unique()
    
    # Store best scores for each concept
    best_scores = []
    
    # For each concept, split data and find best factor
    for concept in concepts:
        concept_data = df[df['concept_id'] == concept]
        
        # Get indices for this concept's data
        indices = concept_data.index.values
        
        # Randomly split indices into train and test
        np.random.seed(41)  # for reproducibility
        train_indices = np.random.choice(indices, size=len(indices)//2, replace=False)
        test_indices = np.array([idx for idx in indices if idx not in train_indices])
        
        # Split data
        train_data = concept_data.loc[train_indices]
        test_data = concept_data.loc[test_indices]
        
        # Find the factor that gives max RuleEvaluator score on train data
        train_rule_scores = train_data['PreferenceVector_RuleEvaluator']
        best_factor = train_data.loc[train_rule_scores.idxmax(), 'factor']
        
        # Get scores for the best factor using test data
        test_factor_data = test_data[test_data['factor'] == best_factor]
        
        if len(test_factor_data) > 0:
            # Get all metrics for this best factor using mean of test data
            metrics_data = {
                'Concept': f'Concept {concept}',
                'Factor': best_factor,
                'Overall': test_factor_data['PreferenceVector_RuleEvaluator'].mean(),
                'Rule Following':  test_factor_data['PreferenceVector_RuleEvaluator_rule_following'].mean(),
                'Relevance': test_factor_data['PreferenceVector_LMJudgeEvaluator_relevance_instruction_ratings'].mean(),
                'Fluency': test_factor_data['PreferenceVector_LMJudgeEvaluator_fluency_ratings'].mean()
            }
            best_scores.append(metrics_data)
    
    # Create DataFrame with best scores
    best_scores_df = pd.DataFrame(best_scores)
    
    # Print the selected factors and their frequencies
    factor_counts = best_scores_df['Factor'].value_counts()
    print("\nSelected Factors Distribution:")
    print(factor_counts)
    
    # Melt the DataFrame for plotting
    plot_df = pd.melt(best_scores_df, 
                      id_vars=['Concept', 'Factor'],
                      value_vars=['Overall', 'Rule Following', 'Relevance', 'Fluency'],
                      var_name='Metric',
                      value_name='Score')

    
    # Print summary statistics
    print("\nSummary Statistics for Test Set Scores:")
    summary_stats = plot_df.groupby('Metric')['Score'].agg(['mean', 'std', 'min', 'max'])
    print(summary_stats)
    
    return p



def plot_best_factor_scores_all(data_path, write_to_path=None, report_to=[], wandb_name=None, mode=None, rule=True):
    # Read the data
    df = pd.read_parquet(data_path)
    
    def harmonic_mean(scores):
        # Return 0 if any score is 0 to maintain strict evaluation
        if 0 in scores:
            return 0
        return len(scores) / sum(1/s for s in scores)
    
    methods = [a[:len(a)-len("_LMJudgeEvaluator_relevance_concept_ratings")] for a in list(df.columns) if "LMJudgeEvaluator_relevance_concept_ratings" in a]

    # Metrics to plot
    if not rule:
        metrics = ['_LMJudgeEvaluator',
                '_LMJudgeEvaluator_relevance_concept_ratings', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Relevance Concept', 'Relevance Instruction', 'Fluency']
    else:
        # Calculate harmonic mean for each method and add as a new column
        for method in methods:
            rule_metric = method + '_RuleEvaluator_rule_following'
            instruction_metric = method + '_LMJudgeEvaluator_relevance_instruction_ratings'
            fluency_metric = method + '_LMJudgeEvaluator_fluency_ratings'
            
            # Create a new column for the harmonic mean
            df[method + '_RuleEvaluator'] = df.apply(
                lambda row: harmonic_mean([
                    row[rule_metric] if ('dataset_name' in df.columns and row['dataset_name'] == 'AlpacaEval' and rule_metric in df.columns and not pd.isna(row[rule_metric])) 
                    else (2-row[rule_metric] if rule_metric in df.columns and not pd.isna(row[rule_metric]) else 0),
                    row[instruction_metric] if instruction_metric in df.columns and not pd.isna(row[instruction_metric]) else 0,
                    row[fluency_metric] if fluency_metric in df.columns and not pd.isna(row[fluency_metric]) else 0
                ]),
                axis=1
            )
            print(f"Scores for {method}:")
            print(df[method + '_RuleEvaluator'])
       
        metrics = [
                '_RuleEvaluator',
                '_RuleEvaluator_rule_following', 
                '_LMJudgeEvaluator_relevance_instruction_ratings', 
                '_LMJudgeEvaluator_fluency_ratings']
        metrics_names = ['Overall', 'Rule Following', 'Relevance Instruction', 'Fluency']
    
    # Get all concepts
    concepts = df['concept_id'].unique()
    
    # Store best scores for each concept
    best_scores = []
    
    # For each concept, find best factor using all data
    for concept in concepts:
        concept_data = df[df['concept_id'] == concept]
        
        # Find the factor that gives max RuleEvaluator score
        rule_scores = concept_data.groupby('factor')['PreferenceVector_RuleEvaluator'].mean()
        best_factor = rule_scores.idxmax()
        
        # Get scores for the best factor
        best_factor_data = concept_data[concept_data['factor'] == best_factor]
        
        # Get all metrics for this best factor
        metrics_data = {
            'Concept': f'Concept {concept}',
            'Factor': best_factor,
            'Overall': best_factor_data['PreferenceVector_RuleEvaluator'].mean(),
            'Rule Following': best_factor_data['PreferenceVector_RuleEvaluator_rule_following'].mean(),
            'Relevance': best_factor_data['PreferenceVector_LMJudgeEvaluator_relevance_instruction_ratings'].mean(),
            'Fluency': best_factor_data['PreferenceVector_LMJudgeEvaluator_fluency_ratings'].mean()
        }
        best_scores.append(metrics_data)
    
    # Create DataFrame with best scores
    best_scores_df = pd.DataFrame(best_scores)

    concept_list = []
    
        # Print detailed statistics for each concept
    print("\nDetailed Statistics for Each Concept:")
    for concept in best_scores_df['Concept'].unique():
        concept_data = best_scores_df[best_scores_df['Concept'] == concept]
        print(f"\n{concept}:")
        print(f"Selected Factor: {concept_data['Factor'].iloc[0]}")
        print("Metrics:")
        for metric in ['Overall', 'Rule Following', 'Relevance', 'Fluency']:
            print(f"  {metric}: {concept_data[metric].iloc[0]:.4f}")
        concept_list.append(concept_data['Factor'].iloc[0] * -1)
            
    # Print the selected factors and their frequencies
    factor_counts = best_scores_df['Factor'].value_counts()
    print("\nSelected Factors Distribution:")
    print(factor_counts)
    print(concept_list)
    
    # Melt the DataFrame for plotting
    plot_df = pd.melt(best_scores_df, 
                      id_vars=['Concept', 'Factor'],
                      value_vars=['Overall', 'Rule Following', 'Relevance', 'Fluency'],
                      var_name='Metric',
                      value_name='Score')
    
   
    # Print summary statistics
    print("\nSummary Statistics for Scores:")
    summary_stats = plot_df.groupby('Metric')['Score'].agg(['mean', 'std', 'min', 'max'])
    print(summary_stats)
