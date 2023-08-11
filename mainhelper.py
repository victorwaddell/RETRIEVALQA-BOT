# Helper functions for main.py

import traceback

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


def get_query():
    query = input("Ask a question! (type 'quit', 'q', or 'exit' to quit): ")
    if len(query) > 1500:  # Prevent excessively long input
        print("Your input is too long. Please try again.")
        return get_query()
    else:
        return str(query)


def get_response(query, chattype):  # Choose response type based on user needs
    try:
        return "Result: " + chattype.run(query)  # Returns answer
    except Exception as e:  
        print(f"An error occurred while getting the answer: {str(e)}")
        traceback.print_exc()
        return None