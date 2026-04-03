from __future__ import annotations

import io
import os
import textwrap
import warnings
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

CHROMA_DIR = Path("chroma_db")
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """Vous etes un assistant pedagogique specialise en analyse documentaire.
Repondez uniquement a partir du contexte fourni.
Si la reponse n'apparait pas dans le contexte, dites clairement que vous ne trouvez pas l'information dans les documents.
N'inventez aucune information.

Contexte :
{context}

Question :
{question}

Reponse en francais, claire et structuree :
"""
)


def clean_text(text: str) -> str:
    """Nettoie grossierement le texte extrait d'un PDF."""
    return " ".join(text.split())


def init_session_state() -> None:
    """Initialise les variables de session utilisees par l'application."""
    defaults = {
        "retriever": None,
        "vector_store": None,
        "collection_name": None,
        "chat_history": [],
        "indexed_files": [],
        "chunk_count": 0,
        "page_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_api_key() -> None:
    """Charge le fichier .env et verifie la presence de la cle OpenAI."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY est introuvable. Ajoutez-la dans le fichier .env avant de lancer l'application."
        )


def get_llm() -> ChatOpenAI:
    """Retourne le modele de generation utilise dans l'application."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_embeddings() -> OpenAIEmbeddings:
    """Retourne le modele d'embeddings utilise pour l'indexation."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def extract_text_from_uploaded_pdfs(uploaded_files: list) -> list[Document]:
    """Extrait le texte des PDF charges et retourne une liste de documents LangChain."""
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        if not pdf_bytes:
            continue

        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": uploaded_file.name, "page": page_number},
                )
            )

    return documents


def split_text_into_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Decoupe les documents en chunks compatibles avec une indexation vectorielle."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list[Document]) -> tuple[Chroma, str]:
    """Construit une base vectorielle Chroma a partir des chunks."""
    if not chunks:
        raise ValueError("Aucun chunk n'est disponible pour l'indexation.")

    CHROMA_DIR.mkdir(exist_ok=True)
    collection_name = f"tp2_rag_streamlit_{uuid4().hex[:8]}"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.persist()
    return vector_store, collection_name


def build_context(docs: list[Document]) -> str:
    """Construit le contexte textuel qui sera envoye au LLM."""
    if not docs:
        return "Aucun extrait pertinent n'a ete retrouve."

    context_parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Extrait {index} | source={source} | page={page}]\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def answer_question(
    question: str,
    retriever,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Execute le retrieval puis la generation d'une reponse."""
    if retriever is None:
        raise ValueError("Le retriever n'est pas initialise. Lancez d'abord l'indexation.")

    llm = llm or get_llm()
    retrieved_docs = retriever.invoke(question)
    context = build_context(retrieved_docs)
    messages = PROMPT_TEMPLATE.format_messages(context=context, question=question)
    answer = llm.invoke(messages).content

    return {
        "question": question,
        "answer": answer,
        "documents": retrieved_docs,
        "context": context,
    }


def render_sources(docs: list[Document]) -> None:
    """Affiche les sources retrouvees dans un expander."""
    with st.expander("Afficher les passages sources", expanded=False):
        if not docs:
            st.info("Aucune source n'a ete retrouvee pour cette question.")
            return

        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "inconnu")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Source {index}** - `{source}` | page `{page}`")
            st.caption(textwrap.shorten(doc.page_content, width=500, placeholder=" ..."))


def render_chat_history() -> None:
    """Affiche l'historique minimal de conversation."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


def main() -> None:
    st.set_page_config(
        page_title="TP2 RAG - Chatbot Streamlit",
        layout="wide",
    )

    init_session_state()

    try:
        ensure_api_key()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.title("Partie 2 - Chatbot RAG avec Streamlit")
    st.write(
        "Cette application permet de charger des PDF, de les indexer dans Chroma "
        "et de poser des questions repondues uniquement a partir du contexte retrouve."
    )

    with st.sidebar:
        st.header("Configuration")
        uploaded_files = st.file_uploader(
            "Charger un ou plusieurs fichiers PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )
        k_value = st.slider("Nombre de chunks recuperes (k)", min_value=1, max_value=8, value=4)
        start_indexing = st.button("Lancer l'indexation", type="primary", use_container_width=True)

        if not uploaded_files:
            st.info("Aucun document charge pour le moment.")

    if start_indexing:
        if not uploaded_files:
            st.warning("Veuillez charger au moins un fichier PDF avant de lancer l'indexation.")
        else:
            try:
                with st.spinner("Extraction du texte, decoupage et indexation en cours..."):
                    documents = extract_text_from_uploaded_pdfs(uploaded_files)
                    if not documents:
                        raise ValueError(
                            "Aucun texte exploitable n'a pu etre extrait des PDF fournis."
                        )

                    chunks = split_text_into_chunks(documents)
                    if not chunks:
                        raise ValueError("Le decoupage n'a produit aucun chunk exploitable.")

                    vector_store, collection_name = build_vector_store(chunks)
                    retriever = vector_store.as_retriever(search_kwargs={"k": k_value})

                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.session_state.collection_name = collection_name
                    st.session_state.indexed_files = [uploaded_file.name for uploaded_file in uploaded_files]
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.page_count = len(documents)
                    st.session_state.chat_history = []

                st.success("Indexation terminee avec succes.")
            except Exception as exc:
                st.error(f"Erreur pendant l'indexation : {exc}")

    if st.session_state.vector_store is not None:
        st.session_state.retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={"k": k_value}
        )

    if st.session_state.retriever is None:
        st.warning("Chargez des PDF puis cliquez sur 'Lancer l'indexation' pour activer le chatbot.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents indexes", len(st.session_state.indexed_files))
        col2.metric("Pages analysees", st.session_state.page_count)
        col3.metric("Chunks crees", st.session_state.chunk_count)
        st.caption(f"Collection Chroma active : {st.session_state.collection_name}")

    render_chat_history()

    question = st.chat_input("Posez une question sur les documents indexes.")

    if question:
        if st.session_state.retriever is None:
            st.warning("Le chatbot n'est pas pret. Lancez d'abord l'indexation des PDF.")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Generation de la reponse..."):
                    result = answer_question(question, st.session_state.retriever, get_llm())
                st.markdown(result["answer"])
                render_sources(result["documents"])

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["documents"],
                }
            )
        except Exception as exc:
            st.error(f"Erreur pendant la generation de la reponse : {exc}")


if __name__ == "__main__":
    main()
