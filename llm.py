from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import os

from extract import get_profile_context  # IMPORT CONTEXT

# ---- API KEY ----
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

# ---- LLM ----
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

SYSTEM_PROMPT = """
You are a personal AI assistant.

Rules:
- Answer ONLY using the provided profile context
- Do NOT guess or infer
- If information is missing, say:
  "I don't have enough information from this profile."
"""

# ---- Chat Function ----
def ask_profile_bot(question, context):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Profile Context:\n{context}"),
        HumanMessage(content=question)
    ]
    # `ChatGroq` is not callable; use `generate` and extract the text.
    # `generate` expects a list of message-lists (batches), so wrap `messages`.
    result = llm.generate([messages])
    try:
        # common structure: result.generations -> List[List[Generation]]
        return result.generations[0][0].text
    except Exception:
        try:
            return result.generations[0].text
        except Exception:
            return str(result)


# ---- RUN ----
if __name__ == "__main__":
    url = "https://kishore8220.netlify.app/"
    context = get_profile_context(url)

    answer = ask_profile_bot(
        "who is kishore?",
        context
    )

    print(answer)
