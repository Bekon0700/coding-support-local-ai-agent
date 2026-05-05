import os
import json
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ─── Config ───────────────────────────────────────────────
DB_PATH = "./db"
EMBED_MODEL = "nomic-embed-text"
# LLM_MODEL = "qwen2.5-coder:7b"
LLM_MODEL = "llama3.1:8b"
# LLM_MODEL = "mistral"
CODEBASE_PATH = "./code-files"
MEMORY_PATH = "./memory.json" 
# ──────────────────────────────────────────────────────────

# Load vector DB once at startup
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)


# ─── Tools ────────────────────────────────────────────────
@tool
def search_codebase(query: str) -> str:
    """Search the codebase for relevant code by meaning.
    Use this when you need to find code related to a concept or feature. DO NOT use this for general knowledge questions.
    """
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
    """Read the complete content of a specific file.
    Use this when you need to see the full content of a file. DO NOT use this for general knowledge questions.
    Input should be the file path like: ./code-files/components/common/header/Header.tsx
    """
    filepath = filepath.strip()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    alt_path = os.path.join(CODEBASE_PATH, filepath)
    if os.path.exists(alt_path):
        with open(alt_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"File not found: {filepath}. Use list_files to see available files."


@tool
def list_files(query: str) -> str:
    """List all available files in the codebase.
    Use this when you need to know what files exist. DO NOT use this for general knowledge questions.
    """
    all_files = []
    for root, dirs, files in os.walk(CODEBASE_PATH):
        dirs[:] = [d for d in dirs if d not in {
            "node_modules", ".next", ".git", ".vscode"
        }]
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in {".ts", ".tsx", ".js", ".jsx", ".json", ".css", ".mjs", ".md"}:
                all_files.append(os.path.join(root, f))
    return "\n".join(all_files)


# ─── Main ─────────────────────────────────────────────────
def save_memory(chat_history):
    """Save conversation to disk."""
    serialized = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serialized.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            serialized.append({"role": "system", "content": msg.content})

    with open(MEMORY_PATH, "w") as f:
        json.dump(serialized, f, indent=2)


def load_memory():
    """Load conversation history from disk."""
    if not os.path.exists(MEMORY_PATH):
        return []

    with open(MEMORY_PATH, "r") as f:
        serialized = json.load(f)

    history = []
    for msg in serialized:
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            history.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            history.append(SystemMessage(content=msg["content"]))
    return history


def trim_history(chat_history, max_messages=20):
    """
    Keep only the last N messages so we don't overflow
    the model's context window in long conversations.
    Always keep the system message if present.
    """
    system_msgs = [m for m in chat_history if isinstance(m, SystemMessage)]
    other_msgs = [m for m in chat_history if not isinstance(m, SystemMessage)]

    # Keep last max_messages
    trimmed = other_msgs[-max_messages:]
    return system_msgs + trimmed

def is_code_question(q: str) -> bool:
    keywords = keywords = [
        "code", "function", "component", "api", "file",
        "hook", "state", "props", "nextjs", "typescript",
        "bug", "error", "import", "export", "tsx", "ts",
        "page", "layout", "header", "footer", "navbar",
        "codebase", "project", "folder", "class", "interface",
        "read", "show me", "what does", "how does", "explain"
    ]
    return any(k in q.lower() for k in keywords)


def main():
    print("⏳ Loading model and vector DB...")

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    tools = [search_codebase, read_file, list_files]

    # System message — gives the agent a persistent identity
    system_message = SystemMessage(content="""
You are an expert code assistant helping a developer understand 
their Next.js TypeScript codebase.

You also have general knowledge. You are allowed to answer directly if you already know the answer.
Tools are optional, not mandatory.

IMPORTANT RULES:
- If the question is about the codebase → use tools
- If the question is general knowledge → answer directly
- If the question is about conversation history → use memory
- Do NOT force tool usage when not needed
- Never describe tool calls in your answer
""")

    agent = create_react_agent(llm, tools)

    print("=" * 50)
    print("🤖 Code Assistant — Agent + Memory")
    print("   Type 'exit' to quit")
    print("   Type 'clear' to wipe memory")
    print("   Type 'history' to see past messages")
    print("=" * 50)

    # Load memory from disk
    chat_history = load_memory()

    if chat_history:
        print(f"\n💾 Loaded {len(chat_history)} messages from previous session.")
    else:
        print("\n🆕 Starting fresh session.")

    # Always prepend system message
    if not any(isinstance(m, SystemMessage) for m in chat_history):
        chat_history.insert(0, system_message)

    print()

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        # Special commands
        if question.lower() == "exit":
            save_memory(chat_history)
            print("💾 Memory saved. Bye!")
            break

        if question.lower() == "clear":
            chat_history = [system_message]
            if os.path.exists(MEMORY_PATH):
                os.remove(MEMORY_PATH)
            print("🗑️  Memory cleared.\n")
            continue

        if question.lower() == "history":
            print("\n── Conversation History ────────────────────")
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    print(f"  You : {msg.content[:100]}")
                elif isinstance(msg, AIMessage):
                    print(f"  AI  : {msg.content[:100]}...")
            print("────────────────────────────────────────────\n")
            continue

        # Add user message to history
        chat_history.append(HumanMessage(content=question))

        # Trim if too long
        chat_history = trim_history(chat_history, max_messages=20)
        
        # 🧠 ROUTING DECISION
        if not is_code_question(question):
            print("\n🧠 Answering directly (no tools)...\n")

            response = llm.invoke(chat_history)
            final_answer = response.content if hasattr(response, "content") else str(response)

            chat_history.append(AIMessage(content=final_answer))
            save_memory(chat_history)

            print("── Final Answer ────────────────────────────")
            print(f"\n{final_answer}\n")
            print("=" * 50 + "\n")
            continue  # ⬅️ VERY IMPORTANT: skip agent

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
                            print(f"🔧 Tool call : {tc['name']}")
                            print(f"   Input     : {tc['args']}\n")
                    elif hasattr(msg, "name") and msg.name:
                        content_preview = str(msg.content)[:300]
                        print(f"📄 Tool result ({msg.name}):")
                        print(f"   {content_preview}...")
                        print()
                    elif hasattr(msg, "content") and msg.content:
                        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                            final_answer = msg.content

        # Add AI response to history
        chat_history.append(AIMessage(content=final_answer))

        # Auto-save after every message
        save_memory(chat_history)

        print("── Final Answer ────────────────────────────")
        print(f"\n{final_answer}\n")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()