import os
import shutil
import tempfile
import pickle

import faiss
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

load_dotenv()

# --- Load the heavy pieces once at startup ---
print("[INFO] Starting up — loading embedding model + LLM...")
EMB = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
LLM = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

MAX_FREE = 15
WELCOME = {"role": "assistant", "content": "Your document is loaded. Ask anything about it."}
LIMIT_MSG = "You've used your 15 free messages.\n\nUnlock full access: calendly.com/yourname"

SYSTEM = (
    "You are a helpful assistant for the user's documents. "
    "Use the provided context when it answers the question. "
    "If the context does not contain the answer, start with "
    "\"This isn't in your documents, but generally:\" and then answer from general knowledge. "
    "Keep answers clear and concise."
)


# ---------- Indexing (in memory, nothing saved to disk) ----------
def build_index(file_paths):
    tmp = tempfile.mkdtemp()
    try:
        for p in file_paths:
            shutil.copy(p, tmp)
        docs = load_all_documents(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # delete uploaded copies — no storage
    chunks = EMB.chunk_documents(docs)
    vecs = np.array(EMB.embed_chunks(chunks)).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    return {"index": index, "texts": [c.page_content for c in chunks]}


def load_sample():
    index = faiss.read_index("faiss_store/faiss.index")
    with open("faiss_store/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return {"index": index, "texts": [m.get("text", "") for m in meta]}


def retrieve(session, question, k=8):
    qv = np.array(EMB.model.encode([question])).astype("float32")
    _, idxs = session["index"].search(qv, k)
    return [session["texts"][i] for i in idxs[0] if 0 <= i < len(session["texts"])]


def build_messages(prior_history, context, question):
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in prior_history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return msgs


def counter_label(count):
    remaining = max(MAX_FREE - count, 0)
    return f"<div id='crux-counter'>{remaining} free messages remaining</div>"


# ---------- UI handlers ----------
def on_upload(file_paths):
    if not file_paths:
        return gr.update(), gr.update(), None, [WELCOME], counter_label(0), 0
    if len(file_paths) > 5:
        gr.Warning("Maximum 5 documents per session.")
        file_paths = file_paths[:5]
    try:
        session = build_index(file_paths)
    except Exception as e:
        gr.Warning(f"Could not read that file: {e}")
        return gr.update(visible=True), gr.update(visible=False), None, [WELCOME], counter_label(0), 0
    return gr.update(visible=False), gr.update(visible=True), session, [WELCOME], counter_label(0), 0


def on_sample():
    session = load_sample()
    return gr.update(visible=False), gr.update(visible=True), session, [WELCOME], counter_label(0), 0


def respond(message, history, session, count):
    """Streaming + memory: types the answer out token-by-token, remembers the conversation."""
    message = (message or "").strip()
    if not message or session is None:
        yield "", history, count, counter_label(count)
        return

    if count >= MAX_FREE:
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": LIMIT_MSG}]
        yield "", history, count, counter_label(count)
        return

    prior = list(history)  # conversation so far (for memory)
    context = "\n\n".join(retrieve(session, message))
    messages = build_messages(prior, context, message)

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    new_count = count + 1
    try:
        partial = ""
        for chunk in LLM.stream(messages):
            partial += (chunk.content or "")
            history[-1]["content"] = partial
            yield "", history, new_count, counter_label(new_count)
    except Exception as e:
        history[-1]["content"] = f"Something went wrong: {e}"
        yield "", history, new_count, counter_label(new_count)


# ---------- Look: deep dark blue, minimal ----------
theme = gr.themes.Base(primary_hue="indigo", neutral_hue="slate").set(
    body_background_fill="#0D1117",
    block_background_fill="#161B22",
    block_border_color="#30363D",
    input_background_fill="#161B22",
    input_border_color="#30363D",
    button_primary_background_fill="#E6E6F0",
    button_primary_text_color="#0D1117",
    button_secondary_background_fill="#21262D",
    button_secondary_text_color="#E6E6F0",
)

css = """
.gradio-container {max-width: 680px !important; margin: 0 auto !important; background: #0D1117 !important;}
#crux-header {text-align: center; padding: 44px 0 6px;}
#crux-header h1 {font-size: 2.7rem; font-weight: 800; color: #F0F0F5; margin: 0; letter-spacing: -0.02em;}
#crux-header p {color: #8B949E; margin: 8px 0 0; font-size: 0.95rem;}
#crux-trust {text-align: center; color: #8B949E; font-size: 0.85rem; margin-top: 14px;}
#crux-badges {text-align: center; margin-top: 10px;}
#crux-badges span {display: inline-block; border: 1px solid #30363D; color: #C9D1D9;
    border-radius: 999px; padding: 3px 12px; margin: 0 4px; font-size: 0.74rem;}
#crux-counter {text-align: right; color: #6E7681; font-size: 0.78rem; padding-top: 6px;}
#crux-cal {text-align: center; margin-top: 14px; font-size: 0.85rem;}
#crux-cal a {color: #A5B4FC; text-decoration: none;}
footer {display: none !important;}
"""

with gr.Blocks(title="Crux") as demo:
    session = gr.State(None)
    count = gr.State(0)

    gr.HTML('<div id="crux-header"><h1>Crux</h1><p>Question in, grounded answer out.</p></div>')

    with gr.Column(visible=True) as upload_view:
        files = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt", ".xlsx"],
            label="Drop your documents here",
        )
        gr.HTML('<div id="crux-trust">No signup. Your session stays until you clear it.</div>')
        gr.HTML('<div id="crux-badges"><span>PDF</span><span>DOCX</span><span>TXT</span><span>Excel</span></div>')
        sample_btn = gr.Button("Try a sample", variant="secondary", size="sm")

    with gr.Column(visible=False) as chat_view:
        chatbot = gr.Chatbot(value=[WELCOME], height=440, show_label=False, avatar_images=(None, None))
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question about your documents...",
                show_label=False, scale=8, container=False, autofocus=True,
            )
            send = gr.Button("Ask", variant="primary", scale=1, min_width=90)
        counter_display = gr.HTML(counter_label(0))
        gr.HTML('<div id="crux-cal">Unlock full access → <a href="https://calendly.com/yourname">calendly.com/yourname</a></div>')

    up_outputs = [upload_view, chat_view, session, chatbot, counter_display, count]
    files.upload(on_upload, [files], up_outputs)
    sample_btn.click(on_sample, None, up_outputs)

    send.click(respond, [msg, chatbot, session, count], [msg, chatbot, count, counter_display], show_progress="hidden")
    msg.submit(respond, [msg, chatbot, session, count], [msg, chatbot, count, counter_display], show_progress="hidden")


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=theme, css=css)
