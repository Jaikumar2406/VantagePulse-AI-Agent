import os
import random

def get_groq_key():
    """
    Returns a random Groq API key from the environment variables to avoid rate limits.
    """
    keys = [
        os.getenv("agent1_llm"),
        os.getenv("agent2_llm"),
        os.getenv("agent3_llm"),
        os.getenv("agent6_llm"),
        os.getenv("agent7_llm"),
        os.getenv("combine_llm")
    ]
    # Filter out None or empty values
    valid_keys = [k for k in keys if k]
    
    if not valid_keys:
        return None
        
    return random.choice(valid_keys)
