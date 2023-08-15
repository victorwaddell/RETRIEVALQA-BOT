def get_user_prompt():
    valid_prompts = [
        'strictlocal_chatagent', 
        'local_chatagent',
        'openlocal_chatagent',
        'general_tutor',
        'idea_generation',
        'writing_bot',
        'debate_bot',
        'factcheck_bot'
    ]
    prompt_message = "Select a chat prompt (type one of the following):\n" + '\n'.join(valid_prompts) + "\n\n"
    
    while True:  # Infinite loop that breaks only when a valid input is given
        user_input = input(prompt_message)
        if user_input in valid_prompts:
            return user_input
        else:
            print("Invalid input. Please select a valid chat prompt.")
