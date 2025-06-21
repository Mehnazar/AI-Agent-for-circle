# circle_agent/main.py
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ✅ Main function with circle assistant
async def main():
    agent = Agent(
        name="CircleMathAssistant",
        instructions=(
            "You are a helpful math tutor who specializes in geometry. "
            "When the user asks about anything related to circles (e.g., circumference, area, diameter, radius), "
            "you give a clear, correct, and simple explanation or formula."
        ),
        model=model
    )

    print("🔵 Circle Math Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("❓ Your question about circle: ")

        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting... Thank You!")
            break

        result = await Runner.run(agent, user_input, run_config=config)
        print("\n📘 Answer:")
        print(result.final_output)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
