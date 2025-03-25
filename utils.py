import secrets

def generate_random_seed(max_seed:int=1000000)->int:
    return secrets.randbelow(max_seed)