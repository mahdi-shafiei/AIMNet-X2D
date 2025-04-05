#hyperparameter.py

# Standard libraries
import random
import itertools
import math

def _sample_hparam_value(hparam_config):
    """Samples a random value based on the hyperparameter configuration."""
    if isinstance(hparam_config, list):
        # Grid search: return a random element from the list
        return random.choice(hparam_config)
    elif isinstance(hparam_config, dict):
        # Random search: sample based on type and range
        if hparam_config.get("type") == "int":
            return random.randint(hparam_config["min"], hparam_config["max"])
        elif hparam_config.get("type") == "float":
            if hparam_config.get("log", False):
                # Sample on a logarithmic scale
                log_min = math.log(hparam_config["min"])
                log_max = math.log(hparam_config["max"])
                log_value = random.uniform(log_min, log_max)
                return math.exp(log_value)
            else:
                return random.uniform(hparam_config["min"], hparam_config["max"])
        elif hparam_config.get("type") == "choice":
            return random.choice(hparam_config["values"])
        else:
            # If it's a dict without type field, just return it as is
            return hparam_config
    else:
        # Fixed value (not a list or dict)
        return hparam_config


