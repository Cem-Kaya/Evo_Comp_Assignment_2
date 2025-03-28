import datetime
import pickle
import secrets

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

   