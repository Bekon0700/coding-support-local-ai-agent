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

print("⏳ Loading vector DB...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
print("✅ Vector DB loaded.")


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
            if ext in {".ts", ".tsx", ".js", ".jsx", ".json", ".css", ".mjs", ".md"}:
                all_files.append(os.path.join(root, f))
    return "\n".join(all_files)


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


def trim_history(chat_history, max_messages=20):
    system_msgs = [m for m in chat_history if isinstance(m, SystemMessage)]
    other_msgs = [m for m in chat_history if not isinstance(m, SystemMessage)]
    return system_msgs + other_msgs[-max_messages:]


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


def chat(user_message, gradio_history):
    lc_history = load_memory()
    if not any(isinstance(m, SystemMessage) for m in lc_history):
        lc_history.insert(0, SYSTEM_MESSAGE)
    lc_history.append(HumanMessage(content=user_message))
    lc_history = trim_history(lc_history)

    tool_calls_made = []
    final_answer = ""

    if not is_code_question(user_message):
        response = llm.invoke(lc_history)
        final_answer = response.content
    else:
        for step in agent.stream({"messages": lc_history}, stream_mode="updates"):
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

    lc_history.append(AIMessage(content=final_answer))
    save_memory(lc_history)

    if tool_calls_made:
        tools_used = " · ".join(set(tool_calls_made))
        final_answer += f"\n\n`🔧 {tools_used}`"

    return final_answer


# ─── Custom CSS ───────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0d0d0d !important;
    font-family: 'Sora', sans-serif !important;
    color: #e8e8e8 !important;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Header */
.app-header {
    padding: 32px 24px 16px;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 8px;
}

.app-title {
    font-size: 18px;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.app-subtitle {
    font-size: 12px;
    color: #555;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

.model-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #888;
}

/* Chatbot */
#chatbot {
    background: #0d0d0d !important;
    border: none !important;
    border-radius: 0 !important;
}

#chatbot .wrap {
    padding: 8px 24px !important;
    gap: 24px !important;
}

/* Messages */
#chatbot .message {
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    max-width: 85% !important;
    box-shadow: none !important;
}

#chatbot .user {
    background: #1c1c1c !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    margin-left: auto !important;
}

#chatbot .bot {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    color: #d4d4d4 !important;
}

/* Code blocks in messages */
#chatbot code {
    background: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: #a8d8a8 !important;
}

#chatbot pre {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    padding: 14px !important;
    overflow-x: auto !important;
}

/* Input area */
.input-area {
    padding: 16px 24px 24px;
    border-top: 1px solid #1a1a1a;
    background: #0d0d0d;
    position: sticky;
    bottom: 0;
}

#msg-input textarea {
    background: #141414 !important;
    border: 1px solid #262626 !important;
    border-radius: 12px !important;
    color: #e8e8e8 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}

#msg-input textarea:focus {
    border-color: #3a3a3a !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.03) !important;
}

#msg-input label {
    display: none !important;
}

/* Send button */
#send-btn {
    background: #f0f0f0 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    font-family: 'Sora', sans-serif !important;
    height: 48px !important;
    transition: background 0.2s, transform 0.1s !important;
    cursor: pointer !important;
}

#send-btn:hover {
    background: #ffffff !important;
    transform: translateY(-1px) !important;
}

#send-btn:active {
    transform: translateY(0) !important;
}

/* Clear button */
#clear-btn {
    background: transparent !important;
    border: 1px solid #222 !important;
    color: #555 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    border-color: #ff4444 !important;
    color: #ff4444 !important;
    background: rgba(255,68,68,0.05) !important;
}

/* Suggestions */
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px 24px;
}

.suggestion-chip {
    background: #141414;
    border: 1px solid #222;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #666;
    cursor: pointer;
    font-family: 'IBM Plex Mono', monospace;
    transition: all 0.2s;
    white-space: nowrap;
}

.suggestion-chip:hover {
    border-color: #444;
    color: #aaa;
    background: #1a1a1a;
}

/* Status bar */
#status-box textarea {
    background: transparent !important;
    border: none !important;
    color: #444 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

#status-box label { display: none !important; }

/* Hide Gradio footer */
footer { display: none !important; }
.built-with { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
"""

# ─── UI ───────────────────────────────────────────────────
with gr.Blocks(title="Code Assistant", css=custom_css) as demo:

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" 
                 stroke="#888" stroke-width="2">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
            </svg>
            Code Assistant
            <span class="model-badge">llama3.1:8b · local</span>
        </div>
        <div class="app-subtitle">RAG · Agent · Memory · 100% offline</div>
    </div>
    """)

    # Suggestion chips
    gr.HTML("""
    <div class="suggestions">
        <span class="suggestion-chip" 
              onclick="document.querySelector('#msg-input textarea').value=this.innerText;
                       document.querySelector('#msg-input textarea').dispatchEvent(new Event('input'))">
            What does Header.tsx do?
        </span>
        <span class="suggestion-chip"
              onclick="document.querySelector('#msg-input textarea').value=this.innerText;
                       document.querySelector('#msg-input textarea').dispatchEvent(new Event('input'))">
            List all files
        </span>
        <span class="suggestion-chip"
              onclick="document.querySelector('#msg-input textarea').value=this.innerText;
                       document.querySelector('#msg-input textarea').dispatchEvent(new Event('input'))">
            Explain types.ts interfaces
        </span>
        <span class="suggestion-chip"
              onclick="document.querySelector('#msg-input textarea').value=this.innerText;
                       document.querySelector('#msg-input textarea').dispatchEvent(new Event('input'))">
            What pages exist in this app?
        </span>
    </div>
    """)

    # Chatbot
    chatbot = gr.Chatbot(
        label="",
        height=520,
        value=load_display_history(),
        elem_id="chatbot",
        show_label=False,
    )

    # Input area
    with gr.Group(elem_classes="input-area"):
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask anything about your codebase...",
                lines=1,
                max_lines=6,
                scale=5,
                elem_id="msg-input",
                show_label=False,
            )
            send_btn = gr.Button(
                "Send",
                variant="primary",
                scale=1,
                elem_id="send-btn",
            )

        with gr.Row():
            clear_btn = gr.Button(
                "clear memory",
                scale=1,
                elem_id="clear-btn",
            )
            status = gr.Textbox(
                value="ready",
                interactive=False,
                scale=4,
                elem_id="status-box",
                show_label=False,
            )

    # ── Handlers ──
    def respond(message, history):
        if not message.strip():
            return history, "", "ready"
        response = chat(message, history)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        return history, "", "✓ done"

    def clear_memory_fn():
        if os.path.exists(MEMORY_PATH):
            os.remove(MEMORY_PATH)
        return [], "memory cleared"

    msg_input.submit(respond, [msg_input, chatbot], [chatbot, msg_input, status])
    send_btn.click(respond, [msg_input, chatbot], [chatbot, msg_input, status])
    clear_btn.click(clear_memory_fn, outputs=[chatbot, status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6969,
        share=False,
    )