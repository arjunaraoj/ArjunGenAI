#ChatApp code
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PIL import Image

# ─── 1) Load environment ──────────────────────────────────────────────────────────
env_path = Path("D:/GenerativeAI/Structured_Unstructured/.env")
load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(f"Missing OPENAI_API_KEY in {env_path}")

# ─── 2) File paths ────────────────────────────────────────────────────────────────
DATA_DIR   = Path("D:/GenerativeAI/Structured_Unstructured")
CSV_PATH   = DATA_DIR / "sales_data.csv"
JSON_PATH  = DATA_DIR / "sales_data.json"
PDF_PATH   = DATA_DIR / "sales_data.pdf"
INDEX_PATH = DATA_DIR / "sales_data_index"

# ─── 3) Load & vectorize documents ───────────────────────────────────────────────
def load_documents():
    docs = []
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        docs.append(Document(page_content=json.dumps(row.to_dict()),
                             metadata={"source": "csv"}))

    json_loader = JSONLoader(file_path=str(JSON_PATH), jq_schema=".[]", text_content=False)
    for d in json_loader.load():
        docs.append(Document(page_content=json.dumps(d.page_content), metadata=d.metadata))

    pdf_loader = PyPDFLoader(str(PDF_PATH))
    docs.extend(pdf_loader.load())

    return docs

# ─── 4) Initialize QA chain ───────────────────────────────────────────────────────
def init_qa_chain():
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if not INDEX_PATH.exists():
        print("🔄 Creating vector index…")
        docs = load_documents()
        db = FAISS.from_documents(docs, embedding)
        db.save_local(str(INDEX_PATH))
    else:
        print("✅ Loading existing index…")
        db = FAISS.load_local(str(INDEX_PATH), embedding, allow_dangerous_deserialization=True)

    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

try:
    qa_chain = init_qa_chain()
except Exception as e:
    print(f"❌ QA init error: {e}")
    qa_chain = None

# ─── 5) Exposed function ─────────────────────────────────────────────────────────
def ask_question(query: str) -> str:
    if qa_chain is None:
        return "⚠️ System not ready. Check logs."
    try:
        res = qa_chain({"query": query})
        return res.get("result", "⚠️ No result returned.")
    except Exception as e:
        return f"❌ Error: {e}"

# ─── 6) Gradio UI (only when run directly) ───────────────────────────────────────
if __name__ == "__main__":
    interface = gr.Interface(
        fn=ask_question,
        inputs=gr.Textbox(label="Ask Sales Data"),
        outputs=gr.Textbox(label="AI Response"),
        title="📊 Sales Data QA",
        description="Ask questions of your CSV, JSON, PDF sales data via OpenAI + FAISS + LangChain.",
        examples=[
            ["What were our top 3 products last quarter?"],
            ["Show me total sales by region."],
        ]
    )
    # Launch locally; AV will no longer block
    interface.launch()
