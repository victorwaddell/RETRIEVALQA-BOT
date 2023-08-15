# Helper functions for main.py

import traceback


def get_query():
    query = input("Ask a question! (type 'quit', 'q', or 'exit' to quit and see highest rated db and chain): ")
    if query.lower() in ['quit', 'q', 'exit']:  # Check for quit-related inputs
        return None
    if len(query) > 1000:  # Prevent excessively long input
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
