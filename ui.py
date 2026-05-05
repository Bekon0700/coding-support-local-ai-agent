import os
import json
import gradio as gr
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ─── Config ───────────────────────────────────────────────
DB_PATH = "./db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
CODEBASE_PATH = "./code-files"
MEMORY_PATH = "./memory.json"
# ──────────────────────────────────────────────────────────

# ─── Load Vector DB ───────────────────────────────────────
print("⏳ Loading vector DB...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
print("✅ Vector DB loaded.")

# ─── Tools ────────────────────────────────────────────────
@tool
def search_codebase(query: str) -> str:
    """Search the codebase for relevant code by meaning."""
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
    """Read the complete content of a specific file."""
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
    """List all available files in the codebase."""
    all_files = []
    for root, dirs, files in os.walk(CODEBASE_PATH):
        dirs[:] = [d for d in dirs if d not in {
            "node_modules", ".next", ".git", ".vscode"
        }]
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in {".ts", ".tsx", ".js", ".jsx",
                       ".json", ".css", ".mjs", ".md"}:
                all_files.append(os.path.join(root, f))
    return "\n".join(all_files)


# ─── Agent Setup ──────────────────────────────────────────
print("⏳ Loading LLM...")
llm = ChatOllama(model=LLM_MODEL, temperature=0)
tools = [search_codebase, read_file, list_files]
agent = create_react_agent(llm, tools)
print("✅ Agent ready.")

SYSTEM_MESSAGE = SystemMessage(content="""
You are an expert code assistant helping a developer understand 
their Next.js TypeScript codebase.

RULES:
- For codebase questions → use tools
- For general knowledge → answer directly  
- For conversation history → use memory
- Never describe tool calls, just execute them
""")

# ─── Memory helpers ───────────────────────────────────────
def save_memory(chat_history):
    serialized = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serialized.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "ai", "content": msg.content})
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
    return history


def trim_history(chat_history, max_messages=20):
    system_msgs = [m for m in chat_history if isinstance(m, SystemMessage)]
    other_msgs = [m for m in chat_history if not isinstance(m, SystemMessage)]
    return system_msgs + other_msgs[-max_messages:]


# ─── GK detector ──────────────────────────────────────────
def is_code_question(q: str) -> bool:
    keywords = [
        "code", "function", "component", "api", "file",
        "hook", "state", "props", "nextjs", "typescript",
        "bug", "error", "import", "export", "tsx", "ts",
        "page", "layout", "header", "footer", "navbar",
        "codebase", "project", "folder", "class", "interface",
        "read", "show me", "what does", "how does", "explain"
    ]
    return any(k in q.lower() for k in keywords)


# ─── Core chat function ───────────────────────────────────
def chat(user_message, gradio_history):
    """
    gradio_history format: list of [user, assistant] pairs
    We maintain our own langchain history separately.
    """
    # Load memory from disk
    lc_history = load_memory()
    if not any(isinstance(m, SystemMessage) for m in lc_history):
        lc_history.insert(0, SYSTEM_MESSAGE)

    # Add current message
    lc_history.append(HumanMessage(content=user_message))
    lc_history = trim_history(lc_history)

    tool_calls_made = []
    final_answer = ""

    if not is_code_question(user_message):
        # Answer directly without agent
        response = llm.invoke(lc_history)
        final_answer = response.content
    else:
        # Run agent
        for step in agent.stream(
            {"messages": lc_history},
            stream_mode="updates"
        ):
            for node, update in step.items():
                messages = update.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls_made.append(tc['name'])
                    elif hasattr(msg, "content") and msg.content:
                        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                            if not hasattr(msg, "name") or not msg.name:
                                final_answer = msg.content

    # Save to memory
    lc_history.append(AIMessage(content=final_answer))
    save_memory(lc_history)

    # Format response with tool info
    response_text = final_answer
    if tool_calls_made:
        tools_used = ", ".join(set(tool_calls_made))
        response_text += f"\n\n🔧 *Tools used: {tools_used}*"

    return response_text


def clear_memory():
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)
    return [], "🗑️ Memory cleared!"


# ─── Gradio UI ────────────────────────────────────────────
with gr.Blocks(title="Local Code Assistant") as demo:

    gr.Markdown("""
    # 🤖 Local Code Assistant
    **Powered by llama3.1:8b + RAG — running 100% on your machine**
    
    Ask anything about your Next.js codebase. No API keys. No internet.
    """)
    
    # ── State ── load previous messages for display
    def load_display_history():
        if not os.path.exists(MEMORY_PATH):
            return []
        with open(MEMORY_PATH, "r") as f:
            data = json.load(f)
        display = []
        for msg in data:
            if msg["role"] == "human":
                display.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "ai":
                display.append({"role": "assistant", "content": msg["content"]})
        return display

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        value=load_display_history()
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask about your codebase...",
            label="Your question",
            scale=4,
            lines=2
        )
        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Memory", variant="secondary")
        status = gr.Textbox(label="Status", interactive=False, scale=3)

    gr.Markdown("""
    ### 💡 Try these:
    - `What does the Header component do?`
    - `Read the types.ts file and explain all interfaces`
    - `What pages exist in this Next.js app?`
    - `List all files in the codebase`
    """)

    # ── Event handlers ──
    def respond(message, history):
        if not message.strip():
            return history, ""
        response = chat(message, history)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        return history, ""

    def clear_memory_fn():
        if os.path.exists(MEMORY_PATH):
            os.remove(MEMORY_PATH)
        return [], "🗑️ Memory cleared!"

    msg_input.submit(respond, [msg_input, chatbot], [chatbot, msg_input])
    send_btn.click(respond, [msg_input, chatbot], [chatbot, msg_input])
    clear_btn.click(clear_memory_fn, outputs=[chatbot, status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )