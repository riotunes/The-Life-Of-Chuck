"""
Shared configuration module for The Life of Chuck
Controls the number of generation stages based on user's age
"""

import random

def calculate_num_stages(user_age, rng=None):
    """
    Calculate the number of future stages based on the user's age using a
    probabilistic model instead of a fixed step table.

    The base expectation still follows a life-expectancy curve (around 80
    years), but a random perturbation introduces variability so that a
    20-year-old can occasionally have fewer stages (early death) and an older
    user can occasionally get an extra stage (long life).

    Args:
        user_age (int): Current age of the user
        rng (random.Random, optional): Injected RNG for reproducibility

    Returns:
        int: Number of generation stages to create (clamped 2-8)
    """
    rng = rng or random

    life_expectancy = 80
    min_stages, max_stages = 2, 8

    # Baseline expectation by age bracket (deterministic backbone)
    if user_age < 20:
        base_stages = 8
    elif user_age < 25:
        base_stages = 7
    elif user_age < 30:
        base_stages = 6
    elif user_age < 40:
        base_stages = 6
    elif user_age < 50:
        base_stages = 5
    elif user_age < 60:
        base_stages = 4
    elif user_age < 70:
        base_stages = 3
    else:
        base_stages = max(2, min(3, (life_expectancy - user_age) // 10 + 1))

    # Age-dependent variability: younger ages get wider swings to allow early death
    variability_scale = 0.25 if user_age < 30 else 0.18 if user_age < 50 else 0.12
    noise = round(rng.normalvariate(0, max(0.5, base_stages * variability_scale)))

    # Small bias toward early death for younger users
    if user_age < 35:
        early_death_chance = min(0.35, max(0.08, (35 - user_age) * 0.01))
        if rng.random() < early_death_chance:
            noise -= rng.randint(1, 3)

    # Occasional longevity bump for older users
    if user_age > 55:
        longevity_chance = min(0.25, 0.05 + (user_age - 55) * 0.005)
        if rng.random() < longevity_chance:
            noise += rng.choice([1, 2])

    num_stages = max(min_stages, min(max_stages, base_stages + noise))
    return num_stages


def get_age_increment():
    """
    Get the age increment per stage.
    Default is 10 years per stage.

    Returns:
        int: Years between each stage
    """
    return 10


def read_user_age(user_data_path="user_data.txt"):
    """
    Read the user's age from the user_data.txt file.

    Args:
        user_data_path (str): Path to user_data.txt file

    Returns:
        int: User's age, or None if not found
    """
    try:
        with open(user_data_path, 'r') as f:
            for line in f:
                if line.startswith('AGE:'):
                    age_str = line.split('AGE:')[1].strip()
                    return int(age_str)
    except (FileNotFoundError, ValueError):
        pass

    return None


def get_num_stages_from_user_data(user_data_path="user_data.txt", default_stages=5, rng=None):
    """
    Convenience function to get the number of stages based on user data file.

    Args:
        user_data_path (str): Path to user_data.txt file
        default_stages (int): Default number of stages if age not found

    Returns:
        int: Number of generation stages
    """
    age = read_user_age(user_data_path)
    if age is not None:
        return calculate_num_stages(age, rng=rng)
    else:
        return default_stages
