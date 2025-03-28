import datetime
import pickle
import secrets
import pandas as pd
from scipy import stats

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