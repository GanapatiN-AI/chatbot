import os
import asyncio
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# ----------------------------
# 1. API Keys (replace with yours)
# ----------------------------


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
print("✅ API Keys configured.")

# ----------------------------
# 2. Model constants
# ----------------------------


MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
APP_NAME = "text_qa_chatbot"
USER_ID = "user123"
SESSION_ID = "session123"

# ----------------------------
# 3. Session setup
# ----------------------------
session_service = InMemorySessionService()

# ----------------------------
# 4. Create chatbot agent
# ----------------------------
qa_agent = Agent(
    name="text_qa",
    model=MODEL_GEMINI_2_0_FLASH,
    description="Answers questions based only on the provided reference text answer should be short and concise.",
    instruction=(
        "you are a regulatory and market access intelligence assistant expert."
        "Do simple normal conversation like a chatbot. "
        "You are a helpful assistant. "
        "Use ONLY the reference text given. "
        "If the answer is not in the text, say 'sorry I could not find that in the reference as my knowledge is limited.' "
        "Always respond in clear, meaningful full sentences."
    ),
)

# ----------------------------
# 5. Main chat loop
# ----------------------------
async def run_text_chatbot(reference_text: str):
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    print("💬 AI Text QA Chatbot (type 'exit' to quit)\n")
    print("📄 Reference text loaded.\n")

    runner = Runner(
        agent=qa_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    while True:
        query = input("Type your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Combine user query + reference text
        user_message = f"Reference text:\n{reference_text}\n\nQuestion: {query}"
        content = types.Content(role="user", parts=[types.Part(text=user_message)])

        final_response_text = "Agent did not produce a final response."

        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                break

        print(f"Bot: {final_response_text}\n")


# ----------------------------
# 6. Run chatbot with sample text
# ----------------------------


if __name__ == "__main__":
    file_path = "reason_final.txt"  # Change to your actual file name/path
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reference_text = f.read()
        asyncio.run(run_text_chatbot(reference_text))
    except Exception as e:
        print(f"❌ Error: {e}")
