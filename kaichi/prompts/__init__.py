import utils as U

def load_prompt(prompt):
    return U.load_text(f"prompts/{prompt}.txt")