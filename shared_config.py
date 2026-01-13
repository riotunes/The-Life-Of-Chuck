"""
Shared configuration module for The Life of Chuck
Controls the number of generation stages based on user's age
"""

def calculate_num_stages(user_age):
    """
    Calculate the number of future stages based on user's current age.

    Logic:
    - Younger people have more stages (longer life expectancy ahead)
    - Older people have fewer stages (shorter life expectancy ahead)
    - Uses approximate life expectancy of 80 years as a baseline

    Args:
        user_age (int): Current age of the user

    Returns:
        int: Number of generation stages to create
    """
    # Life expectancy baseline
    life_expectancy = 80

    # Calculate remaining years
    remaining_years = max(life_expectancy - user_age, 10)  # Minimum 10 years

    # Calculate stages based on remaining years
    # Each stage represents roughly 10 years, but we adjust the count
    if user_age < 20:
        # Teens: 7-8 stages (60+ years ahead)
        num_stages = 8
    elif user_age < 25:
        # Early 20s: 7 stages
        num_stages = 7
    elif user_age < 30:
        # Late 20s: 6 stages
        num_stages = 6
    elif user_age < 40:
        # 30s: 5-6 stages
        num_stages = 6
    elif user_age < 50:
        # 40s: 5 stages
        num_stages = 5
    elif user_age < 60:
        # 50s: 4 stages
        num_stages = 4
    elif user_age < 70:
        # 60s: 3 stages
        num_stages = 3
    else:
        # 70+: 2-3 stages
        num_stages = max(2, min(3, (life_expectancy - user_age) // 10 + 1))

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


def get_num_stages_from_user_data(user_data_path="user_data.txt", default_stages=5):
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
        return calculate_num_stages(age)
    else:
        return default_stages
