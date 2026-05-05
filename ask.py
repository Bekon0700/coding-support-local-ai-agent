from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Config ───────────────────────────────────────────────
DB_PATH = "./db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
# ──────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """
You are an expert code assistant helping a developer understand 
their Next.js TypeScript codebase.

Use ONLY the code context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the codebase."
Always mention which file the answer comes from. explain like you are explaining to a donkey. with proper example

Context:
{context}

Question: {question}

Answer:
"""

def format_docs(docs):
    """Format retrieved docs and show their sources."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"--- From: {source} ---\n{doc.page_content}")
    # print("\n\n".join(formatted))
    return "\n\n".join(formatted)


def main():
    print("=" * 50)
    print("🤖 Code Assistant — Ask Mode")
    print("   Type 'exit' to quit")
    print("=" * 50)

    print("\n⏳ Loading vector database...")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",        # MMR = Maximum Marginal Relevance
        search_kwargs={
            "k": 6,               # Return 6 chunks
            "fetch_k": 20,        # But first fetch 20, then pick best 6
            "lambda_mult": 0.7    # Balance relevance vs diversity
        }
    )

    llm = OllamaLLM(model=LLM_MODEL)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Modern LCEL chain (LangChain Expression Language)
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ Ready! Ask me anything about your codebase.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye!")
            break

        print("\n🤔 Thinking...\n")

        # Show retrieved sources before answering
        retrieved_docs = retriever.invoke(question)
        print("📁 Relevant files found:")
        seen = set()
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                print(f"   - {src}")
                seen.add(src)
        print()

        # Stream the answer token by token
        print("Assistant: ", end="", flush=True)
        for token in chain.stream(question):
            print(token, end="", flush=True)

        print("\n\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()