import os
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ─── Config ───────────────────────────────────────────────
DB_PATH = "./db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:7b"
CODEBASE_PATH = "./code-files"
MEMORY_PATH = "./memory-gpt.json"
# ──────────────────────────────────────────────────────────

# Load vector DB
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# ─── Tools ────────────────────────────────────────────────
@tool
def search_codebase(query: str) -> str:
    """Search ONLY the codebase for code-related queries."""
    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        return "No relevant code found."
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        results.append(f"--- File: {source} ---\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def read_file(filepath: str) -> str:
    """Read full file content."""
    filepath = filepath.strip()

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    alt_path = os.path.join(CODEBASE_PATH, filepath)
    if os.path.exists(alt_path):
        with open(alt_path, "r", encoding="utf-8") as f:
            return f.read()

    return f"File not found: {filepath}"


@tool
def list_files(query: str) -> str:
    """List all project files."""
    all_files = []
    for root, dirs, files in os.walk(CODEBASE_PATH):
        dirs[:] = [d for d in dirs if d not in {
            "node_modules", ".next", ".git"
        }]
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in {".ts", ".tsx", ".js", ".jsx", ".json"}:
                all_files.append(os.path.join(root, f))
    return "\n".join(all_files)


# ─── Memory ───────────────────────────────────────────────
def save_memory(chat_history):
    serialized = []
    for msg in chat_history:
        role = "human" if isinstance(msg, HumanMessage) else \
               "ai" if isinstance(msg, AIMessage) else "system"
        serialized.append({"role": role, "content": msg.content})

    with open(MEMORY_PATH, "w") as f:
        json.dump(serialized, f, indent=2)


def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return []

    with open(MEMORY_PATH, "r") as f:
        data = json.load(f)

    history = []
    for msg in data:
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            history.append(AIMessage(content=msg["content"]))
        else:
            history.append(SystemMessage(content=msg["content"]))
    return history


def trim_history(chat_history, max_messages=20):
    system_msgs = [m for m in chat_history if isinstance(m, SystemMessage)]
    others = [m for m in chat_history if not isinstance(m, SystemMessage)]
    return system_msgs + others[-max_messages:]


# ─── Routing Logic ─────────────────────────────────────────
def is_code_question(q: str) -> bool:
    keywords = [
        "code", "function", "component", "api",
        "file", "hook", "error", "bug", "import"
    ]
    return any(k in q.lower() for k in keywords)


def is_puzzle(q: str) -> bool:
    keywords = [
        "puzzle", "riddle", "bridge", "cross",
        "logic", "minutes", "people", "torch"
    ]
    return any(k in q.lower() for k in keywords)


# ─── Main ─────────────────────────────────────────────────
def main():
    print("⏳ Loading...")

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    tools = [search_codebase, read_file, list_files]

    # Agent system prompt
    agent_system = SystemMessage(content="""
You are an expert code assistant.

- Use tools ONLY for code-related queries
- Use chat history for follow-ups
- Do NOT answer general knowledge using tools
""")

    agent = create_react_agent(llm, tools)

    chat_history = load_memory()

    if not any(isinstance(m, SystemMessage) for m in chat_history):
        chat_history.insert(0, agent_system)

    print("🤖 Ready (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            save_memory(chat_history)
            break

        if not question:
            continue

        chat_history.append(HumanMessage(content=question))
        chat_history = trim_history(chat_history)

        # ─── ROUTING ─────────────────────────

        # 🧩 PUZZLE MODE
        if is_puzzle(question):
            print("\n🧠 Puzzle solving...\n")

            messages = [
                SystemMessage(content="""
Solve the puzzle step-by-step.

Rules:
- After each step, calculate total = previous total + step time
- NEVER guess totals
- NEVER skip calculation
- If total is wrong, fix it before continuing

Format:

Step 1: ...
Step time: X
Total: Y

Only proceed if total is correct.

After finishing:
- Recalculate total from scratch
- Ensure it equals final answer
- If not, fix the steps
"""),
                HumanMessage(content=question)
            ]

            response = llm.invoke(messages)
            final_answer = response.content

        # 🌍 GENERAL MODE
        elif not is_code_question(question):
            print("\n🧠 General answer...\n")

            messages = [
                SystemMessage(content="""
You are a helpful assistant.

- Answer directly
- Do NOT mention tools
"""),
                HumanMessage(content=question)
            ]

            response = llm.invoke(messages)
            final_answer = response.content

        # 🧑‍💻 CODE MODE (AGENT)
        else:
            print("\n── Agent Thinking ──────────────────────────\n")

            final_answer = ""

            for step in agent.stream(
                {"messages": chat_history},
                stream_mode="updates"
            ):
                for node, update in step.items():
                    messages = update.get("messages", [])

                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"🔧 Tool: {tc['name']}")
                                print(f"   Input: {tc['args']}\n")

                        elif hasattr(msg, "name") and msg.name:
                            print(f"📄 Tool result ({msg.name})")

                        elif hasattr(msg, "content") and msg.content:
                            final_answer = msg.content

        # Save + print
        chat_history.append(AIMessage(content=final_answer))
        save_memory(chat_history)

        print("\n── Final Answer ────────────────────────────\n")
        print(final_answer)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()