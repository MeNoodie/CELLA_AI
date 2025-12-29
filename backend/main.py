import os
import tempfile
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    temperature=0.1,
    max_output_tokens=300
)


st.set_page_config(
    page_title="Cella Assistant",
    page_icon="🌟",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"]
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = None

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.session_state.document_text = read_pdf(tmp_path)
    os.remove(tmp_path)

    st.sidebar.success(f"{uploaded_file.name} loaded ✅")

st.title("Cella Assistant 🌟", anchor=False)

st.session_state.selected_suggestion = st.pills(
    label="Examples",
    label_visibility="collapsed",
    options=[
        "Tell me about RAG",
        "Help me understand Prompt Engineering",
        "How to make an Agent Chatbot with LangChain",
    ],
)


def clear_conversation():
    st.session_state.messages = []

st.button("Restart Conversation 🔄", on_click=clear_conversation)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt_input = st.chat_input("Ask your question...")

user_query = (
    st.session_state.selected_suggestion
    if st.session_state.selected_suggestion and not prompt_input
    else prompt_input
)


system_template = """
You are an AI assistant that reads and understands PDF documents.

Your task is to answer user questions strictly using the provided document context.

Rules:
1. Answer ONLY the asked question.
2. Use information ONLY from the document context.
3. Keep the answer precise, accurate, and summarized.
4. Answer must be less than 200 words.
5. If answer is not found, say:
   "The answer is not available in the provided document."

Context:
{context}

Conversation rules:
- Greet only in the first response.
- End politely when conversation finishes.
"""


if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.document_text:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("human", "{question}")
            ])

            chain = prompt | llm

            response_obj = chain.invoke({
                "context": st.session_state.document_text[:8000],  # safety limit
                "question": user_query
            })

            answer = response_obj.content

        else:
            answer = llm.invoke(user_query).content

        st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
