import os
from google import genai
from google.genai import types  # Needed for Tool configuration
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Setup Clients
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


# 1. Define the tool function
def search_the_web(query: str):
    """Searches the live web for current information and news."""
    print(f"\n[Gemini is invoking Tavily Search for: {query}]")
    result = tavily.search(query=query, search_depth="basic")

    # Format the results into a readable string for the model
    context = "\n".join([f"- {r['title']}: {r['content']}" for r in result['results']])
    return context


# 2. Define the tools list
# We pass the actual function here
tools=[search_the_web]


def run_agent():
    print("AI Agent is ready. Type 'exit' to quit.")
    chat = client.chats.create(
        model='gemini-2.5-flash-lite',
        config=types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False  # This enables automatic execution of your function
            )
        )
    )
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # The .send_message call will automatically trigger search_the_web
        # if the model thinks it's necessary!
        # In the new SDK, send_message is called on the chat object created above
        response = chat.send_message(user_input)

        print(f"\nAgent: {response.text}")


if __name__ == "__main__":
    run_agent()
