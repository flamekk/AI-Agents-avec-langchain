from __future__ import annotations

import base64
import os
import shutil
import tempfile
import textwrap
import warnings
from pathlib import Path
from uuid import uuid4

import fitz
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pypdf import PdfReader

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

CHROMA_DIR = Path("chroma_db_multimodal")
TEXT_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
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
    """Nettoie grossierement le texte extrait."""
    return " ".join(text.split())


def init_session_state() -> None:
    """Initialise les variables de session necessaires a l'application."""
    defaults = {
        "retriever": None,
        "vector_store": None,
        "collection_name": None,
        "page_images": [],
        "indexed_files": [],
        "text_history": [],
        "vision_history": [],
        "chunk_count": 0,
        "page_count": 0,
        "temp_dir": tempfile.mkdtemp(prefix="tp2_rag_multimodal_"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_api_key() -> None:
    """Charge le fichier .env et verifie la cle API."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY est introuvable. Ajoutez-la dans le fichier .env avant de lancer l'application."
        )


def get_llm() -> ChatOpenAI:
    """Modele utilise pour la generation texte et l'analyse visuelle."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_embeddings() -> OpenAIEmbeddings:
    """Modele d'embeddings utilise pour l'indexation."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def save_uploaded_pdfs(uploaded_files: list, output_dir: Path) -> list[Path]:
    """Sauvegarde les fichiers PDF televerses dans un dossier temporaire."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        target_path = output_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(target_path)

    return saved_paths


def load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    """Charge les PDF en une liste de pages texte."""
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": pdf_path.name, "page": page_number},
                )
            )

    return documents


def split_text_into_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Decoupe le texte pour l'indexation."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list[Document]) -> tuple[Chroma, str]:
    """Construit une base vectorielle Chroma pour le mode texte."""
    if not chunks:
        raise ValueError("Aucun chunk n'est disponible pour l'indexation.")

    CHROMA_DIR.mkdir(exist_ok=True)
    collection_name = f"tp2_rag_multimodal_{uuid4().hex[:8]}"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.persist()
    return vector_store, collection_name


def build_context(docs: list[Document]) -> str:
    """Construit le contexte textuel utilise dans le prompt RAG."""
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


def encode_image(image_path: str | Path) -> str:
    """Encode une image PNG en base64 pour l'envoyer au modele multimodal."""
    image_path = Path(image_path)
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def pdf_pages_to_images(pdf_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Convertit chaque page d'un PDF en image PNG."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    pdf_document = fitz.open(str(pdf_path))

    try:
        for page_index in range(pdf_document.page_count):
            page = pdf_document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1}.png"
            pixmap.save(str(image_path))
            image_paths.append(image_path)
    finally:
        pdf_document.close()

    return image_paths


def normalize_response_content(content) -> str:
    """Normalise les sorties du modele afin d'obtenir une chaine simple."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                text_parts.append(item["text"])
        if text_parts:
            return "\n".join(text_parts).strip()

    return str(content)


def describe_image_with_llm(image_path: str | Path, question: str) -> str:
    """Analyse visuellement une page PDF convertie en image."""
    encoded_image = encode_image(image_path)
    llm = get_llm()

    prompt = (
        "Vous etes un assistant d'analyse visuelle de documents. "
        "Repondez uniquement a partir de ce qui est visible sur l'image fournie. "
        "Si l'information n'est pas lisible ou absente, dites-le clairement.\n\n"
        f"Question : {question}"
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            },
        ]
    )

    response = llm.invoke([message])
    return normalize_response_content(response.content)


def answer_text_question(question: str, retriever) -> dict:
    """Repond a une question textuelle via un pipeline RAG classique."""
    if retriever is None:
        raise ValueError("Le retriever n'est pas initialise.")

    retrieved_docs = retriever.invoke(question)
    context = build_context(retrieved_docs)
    messages = TEXT_PROMPT_TEMPLATE.format_messages(context=context, question=question)
    response = get_llm().invoke(messages)

    return {
        "question": question,
        "answer": normalize_response_content(response.content),
        "documents": retrieved_docs,
        "context": context,
    }


def render_text_sources(docs: list[Document]) -> None:
    """Affiche les sources textuelles retrouvees."""
    with st.expander("Afficher les passages sources", expanded=False):
        if not docs:
            st.info("Aucune source n'a ete retrouvee.")
            return

        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "inconnu")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Source {index}** - `{source}` | page `{page}`")
            st.caption(textwrap.shorten(doc.page_content, width=500, placeholder=" ..."))


def reset_temp_workspace(temp_root: Path) -> Path:
    """Recree un espace temporaire propre pour chaque nouvelle indexation."""
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


def main() -> None:
    st.set_page_config(
        page_title="TP2 RAG - Application Multimodale",
        layout="wide",
    )

    init_session_state()

    try:
        ensure_api_key()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.title("Partie 3 - RAG multimodal")
    st.write(
        "Cette application combine un RAG textuel classique et une analyse visuelle "
        "des pages PDF converties en images."
    )

    with st.sidebar:
        st.header("Configuration")
        uploaded_files = st.file_uploader(
            "Charger un ou plusieurs PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )
        k_value = st.slider("Nombre de chunks recuperes (k)", min_value=1, max_value=8, value=4)
        launch_processing = st.button("Indexer et convertir les pages", type="primary", use_container_width=True)

        if not uploaded_files:
            st.info("Aucun PDF charge pour le moment.")

    if launch_processing:
        if not uploaded_files:
            st.warning("Veuillez charger au moins un fichier PDF avant de lancer le traitement.")
        else:
            try:
                with st.spinner("Traitement des PDF, indexation et conversion des pages..."):
                    temp_root = reset_temp_workspace(Path(st.session_state.temp_dir))
                    pdf_dir = temp_root / "pdfs"
                    image_root = temp_root / "images"

                    saved_pdf_paths = save_uploaded_pdfs(uploaded_files, pdf_dir)
                    documents = load_pdf_documents(saved_pdf_paths)
                    if not documents:
                        raise ValueError(
                            "Aucun texte exploitable n'a pu etre extrait des PDF fournis."
                        )

                    chunks = split_text_into_chunks(documents)
                    if not chunks:
                        raise ValueError("Le decoupage n'a produit aucun chunk exploitable.")

                    vector_store, collection_name = build_vector_store(chunks)
                    retriever = vector_store.as_retriever(search_kwargs={"k": k_value})

                    page_images = []
                    for pdf_path in saved_pdf_paths:
                        pdf_image_dir = image_root / pdf_path.stem
                        image_paths = pdf_pages_to_images(pdf_path, pdf_image_dir)
                        for page_number, image_path in enumerate(image_paths, start=1):
                            page_images.append(
                                {
                                    "pdf_name": pdf_path.name,
                                    "page_number": page_number,
                                    "image_path": str(image_path),
                                }
                            )

                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.session_state.collection_name = collection_name
                    st.session_state.page_images = page_images
                    st.session_state.indexed_files = [uploaded_file.name for uploaded_file in uploaded_files]
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.page_count = len(documents)
                    st.session_state.text_history = []
                    st.session_state.vision_history = []

                st.success("Indexation multimodale terminee avec succes.")
            except Exception as exc:
                st.error(f"Erreur pendant le traitement : {exc}")

    if st.session_state.vector_store is not None:
        st.session_state.retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={"k": k_value}
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("PDF indexes", len(st.session_state.indexed_files))
    col2.metric("Pages texte analysees", st.session_state.page_count)
    col3.metric("Images de pages", len(st.session_state.page_images))

    if st.session_state.collection_name:
        st.caption(f"Collection Chroma active : {st.session_state.collection_name}")

    text_tab, vision_tab = st.tabs(["RAG Texte", "Vision / Pages"])

    with text_tab:
        st.subheader("Questions textuelles sur les documents")
        if st.session_state.retriever is None:
            st.warning("Chargez des PDF puis lancez l'indexation pour activer le RAG texte.")
        else:
            for item in st.session_state.text_history:
                st.markdown(f"**Question :** {item['question']}")
                st.markdown(f"**Reponse :** {item['answer']}")
                render_text_sources(item["documents"])

            with st.form("text_question_form"):
                text_question = st.text_input("Votre question textuelle")
                submit_text_question = st.form_submit_button("Obtenir la reponse")

            if submit_text_question:
                if not text_question.strip():
                    st.warning("Veuillez saisir une question.")
                else:
                    try:
                        with st.spinner("Recherche des passages pertinents et generation de la reponse..."):
                            result = answer_text_question(text_question.strip(), st.session_state.retriever)

                        st.markdown(f"**Question :** {result['question']}")
                        st.markdown(f"**Reponse :** {result['answer']}")
                        render_text_sources(result["documents"])
                        st.session_state.text_history.append(result)
                    except Exception as exc:
                        st.error(f"Erreur pendant la question textuelle : {exc}")

    with vision_tab:
        st.subheader("Analyse visuelle d'une page PDF")
        if not st.session_state.page_images:
            st.warning("Aucune page image n'est disponible. Lancez d'abord le traitement des PDF.")
        else:
            options = [
                f"{item['pdf_name']} - page {item['page_number']}"
                for item in st.session_state.page_images
            ]
            selected_label = st.selectbox("Selectionnez une page a analyser", options=options)
            selected_index = options.index(selected_label)
            selected_page = st.session_state.page_images[selected_index]
            selected_image_path = Path(selected_page["image_path"])

            st.image(
                Image.open(selected_image_path),
                caption=selected_label,
                use_container_width=True,
            )

            with st.expander("Afficher les pages converties", expanded=False):
                for item in st.session_state.page_images:
                    st.image(
                        item["image_path"],
                        caption=f"{item['pdf_name']} - page {item['page_number']}",
                        width=220,
                    )

            for item in st.session_state.vision_history:
                st.markdown(f"**Page analysee :** {item['page_label']}")
                st.markdown(f"**Question :** {item['question']}")
                st.markdown(f"**Reponse :** {item['answer']}")

            with st.form("vision_question_form"):
                vision_question = st.text_input("Votre question sur la page selectionnee")
                submit_vision_question = st.form_submit_button("Analyser la page")

            if submit_vision_question:
                if not vision_question.strip():
                    st.warning("Veuillez saisir une question visuelle.")
                else:
                    try:
                        with st.spinner("Analyse visuelle de la page en cours..."):
                            vision_answer = describe_image_with_llm(
                                selected_image_path,
                                vision_question.strip(),
                            )

                        st.markdown(f"**Question :** {vision_question.strip()}")
                        st.markdown(f"**Reponse :** {vision_answer}")

                        st.session_state.vision_history.append(
                            {
                                "page_label": selected_label,
                                "question": vision_question.strip(),
                                "answer": vision_answer,
                            }
                        )
                    except Exception as exc:
                        st.error(f"Erreur pendant l'analyse visuelle : {exc}")


if __name__ == "__main__":
    main()
