import datetime
import pickle
import secrets
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def generate_random_seed(max_seed:int=1000000)->int:
    return secrets.randbelow(max_seed)

def save_as_pickle(results, experiment_name, folder:str='./pckl'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{folder}/{timestamp}_{experiment_name}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_pickle(filename:str):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_ils_results_from_pickle(filename:str):
    results = load_pickle(filename) #load the results from the pickle file, it is a list of dictionaries.
    summary = results.pop()
    return summary, results

#columns = ['mutation_size', 'initial_cut', 'best_cut_size', 'time_elapsed', 'average_cut_size', 'n_stays_in_local_optimum']
column_names_dict = {
    'mutation_size': 'Mutation Size',
    'initial_cut': 'Initial Cut Size',
    'average_cut_size': 'Avg Cut Size',
    'n_stays_in_local_optimum': 'Stucks in Local Optimum',
    'best_cut_size': 'Best Cut',
    'time_elapsed': 'Time Elapsed',
    'avg_cut_size': 'Avg Cut Size',
    'initial_cut_size_avg': 'Avg Initial Cut Size',
    'initial_cut_size_best': 'Best Initial Cut Size',
    'avg_time_per_fm': 'Avg Time Per FM'}

def convert_results_to_dataframe(results:list[dict],columns=['best_cut_size', 'time_elapsed', 'avg_cut_size', 
    'initial_cut_size_avg', 'avg_time_per_fm'], generate_html=True)->tuple[pd.DataFrame, str]:
    run_data = results[:-1] # exclude the summary
    df = pd.DataFrame(run_data)

    #['initial_cut_size_avg', 'avg_time_per_fm'] not in index
    # pick the columns to display
    df = df[columns]
    #LLM Prompt: get the matching column names from the column_names_dict and rename the columns of dataframe
    #if it exists in the dictionary.
    df.columns = [column_names_dict[col] if col in column_names_dict else col for col in columns]
       
    #LLM Prompt: convert the dataframe to html table.
    # Display as HTML table with styling
    html_table = None
    if generate_html:
        styled_df = df.style.set_properties(**{'text-align': 'center'})
        styled_df = styled_df.format(precision=3)
        styled_df = styled_df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])            
        styled_df = styled_df.hide(axis='index')
        html_table = styled_df.to_html()
    return df, html_table 

def perform_mann_whitney_u_test(values_1:list[int], values_2:list[int]):
    statistic, p_value = stats.mannwhitneyu(values_1, values_2, alternative='two-sided')
    return statistic, p_value

def compact_to_blocks(se:list, n_blocks=5, rnd=2)->tuple[list, list]:
    """Splits each item in se list in n_blocks equal parts and calculates their means. 
    If there is a remainder when splitting, the length of last item will be different than previous ones.
    
    Args:
        se (list): a list of numbers of any size.
        n_blocks (int, optional): Number of desired parts. Cannot be more than shortest item in se list. Defaults to 5.
        rnd (int, optional): Decimals when rounding the means. Defaults to 2.

    Returns:
        tuple[list, list]: (means of blocks, blocks)
    """
    
    if len(se) < n_blocks:
        raise "Length of shortest sequence cannot be less than block count."
    step_size = round(len(se)/n_blocks)
    ph = []    
    means = []
    
    for i in range(0, n_blocks - 1):        
        #ph.append(round(np.mean(se[i:i+step_size]), 2))
        block = se[i*step_size:i*step_size + step_size]
        ph.append(block)
        means.append(round(np.mean(block),rnd))    
    #append the rest
    i+=1
    block = se[i*step_size:]
    ph.append(block)
    means.append(round(np.mean(block),rnd))
        
    recon = []
    for p in ph:
        recon += p
    if len(recon) != len(se):
        pass
    assert recon == se , f"reconstructed sequence {recon} does not match original {se}"
    return means, ph

def generate_mutation_size_trend_chart(phase_means:list):
    """Generates a chart to display trend of adaptive mutation phases. X-axis is the number of phases,
    y-axis is the mutation sizes. Adds a faded line for each phase and a main trend as mean and standard deviation.
    Contains error bars.

    Call this method and call plt.show() t display it.
    Args:
        phase_means (list): means of adaptive mutation size phases. It is basically an array of  equal-sized arrays.
    """
    # LLM: Prompt:list 'phase_means' contain 160 rows. each row is another list of 5 items. 
    # Generate a chart where x-axis is 1-5 and and y-axis values are in column x value-1.
    
    # Convert phase_means to array for easier manipulation
    phase_means_array = np.array(phase_means)

    # Calculate statistics for each phase
    phase_averages = np.mean(phase_means_array, axis=0)
    phase_std = np.std(phase_means_array, axis=0)

    # Create x values (1-5)
    x = np.arange(1, 6)

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot individual sequences (with low opacity)
    for seq in phase_means:
        plt.plot(x, seq, 'o-', alpha=0.05, color='gray')

    # Plot the average with error bars
    plt.errorbar(x, phase_averages, yerr=phase_std, 
                capsize=5, linewidth=2, marker='o', markersize=8, 
                color='blue', label='Average with std deviation')

    # Plot the average line more prominently
    #plt.plot(x, phase_averages, 'o-', linewidth=3, markersize=10, color='red', label='Average')

    # Customize plot
    plt.xlabel('Phase', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title('Phase Means across 160 Sequences', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x)
    plt.legend()

    # Add some statistics as text
    #plt.figtext(0.15, 0.02, f"Phase averages: {[round(val, 2) for val in phase_averages]}", fontsize=10)
    #plt.figtext(0.65, 0.02, f"Standard deviations: {[round(val, 2) for val in phase_std]}", fontsize=10)