import json
import time
from groq import Groq, BadRequestError
from ddgs import DDGS

# gpt-oss-120b is Groq's current recommended general-purpose model and is
# noticeably more reliable at tool calling than llama-3.3-70b-versatile,
# which occasionally emits a malformed function call and errors out.
# See https://console.groq.com/docs/models for the current model list.
MODEL = "openai/gpt-oss-120b"

client = Groq()  # reads GROQ_API_KEY from your environment

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to a web_search tool. "
    "Web search is good for general facts, news, and articles, but it often "
    "cannot find live, transactional data -- e.g. exact bus/train timings for "
    "a specific date, live seat availability, or current ticket prices -- "
    "because that lives inside booking systems, not search-indexed pages. "
    "If your first 1-2 searches don't turn up the specific detail asked for, "
    "STOP searching further variations of the same query. Instead, share "
    "whatever useful general information you did find (e.g. typical "
    "schedules, journey duration, operator names, relevant links), and "
    "clearly tell the user to check the exact live details on the official "
    "source (e.g. the operator's own website/app, or a booking aggregator)."
)


# ---------------------------------------------------------------------------
# STEP 1: Define the tool (Groq uses the same OpenAI-style function schema).
# ---------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current or factual information. Use "
                "this whenever the answer might depend on recent events, "
                "specific numbers, names, or anything you are not fully "
                "certain about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short, specific search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# STEP 2: What the tool actually does. Plain Python, no AI involved.
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in results)


# ---------------------------------------------------------------------------
# STEP 3: The agent loop.
# ---------------------------------------------------------------------------
def run_agent(user_question: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for step in range(max_steps):
        # Occasionally the model emits a malformed tool call and Groq
        # rejects it with a 400 (code "tool_use_failed"). This is usually
        # transient, so retry a couple of times before giving up.
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=1024,
                )
                break
            except BadRequestError as e:
                if "tool_use_failed" in str(e) and attempt < 2:
                    print("  \u26a0\ufe0f  malformed tool call, retrying...")
                    time.sleep(1)
                    continue
                raise
        msg = response.choices[0].message

        if msg.tool_calls:
            # Record the assistant's turn, including which tool(s) it wants
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                if tc.function.name == "web_search":
                    args = json.loads(tc.function.arguments)
                    query = args["query"]
                    print(f"  \U0001F50D searching: {query}")
                    result = web_search(query)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            # loop again so the model can read the results and decide next step
        else:
            return msg.content

    # Step limit reached -- instead of just giving up, ask the model to
    # summarize whatever it did find (tool_choice="none" forces a plain
    # text answer, no more tool calls allowed).
    final = client.chat.completions.create(
        model=MODEL,
        messages=messages
        + [
            {
                "role": "user",
                "content": (
                    "You're out of search attempts. Based on everything "
                    "found so far, give your best available answer, and "
                    "clearly point the user to where they can check exact "
                    "or live details themselves."
                ),
            }
        ],
        tool_choice="none",
        max_tokens=1024,
    )
    return final.choices[0].message.content


# ---------------------------------------------------------------------------
# STEP 4: A tiny command-line chat loop to try it out.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Research Agent (Groq) -- ask me anything (type 'quit' to exit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = run_agent(question)
        print(f"\nAgent: {answer}\n")