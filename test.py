from backend import run_travel_agent

user_input = input("Enter your travel request: ")

response = run_travel_agent(user_input, 'test_thread_id')

print(response["answer"])