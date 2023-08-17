# Helper functions for main.py

import traceback


def get_query():
    query = input("Ask a question!: ")
    if 0 < len(query.strip()) <= 1000:  # Valid input
        return str(query)
    else:  # Invalid input
        print("Your input is invalid. Please try again.")
        return get_query()


def get_response(query, chattype):  # Choose response type based on user needs
    try:
        return "Result: " + chattype.run(query)  # Returns answer
    except Exception as e:  
        print(f"An error occurred while getting the answer: {str(e)}")
        traceback.print_exc()
        return None
