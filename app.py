
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "langchain_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="LangChain RAG - Search",
    page_icon="🔍",
    layout="wide"
)

st.title("LangChain RAG - Retrieval Explorer")

st.write(
    "Search through LangChain documentation "
    "using semantic similarity."
)

# -----------------------------
# Load Embedding Model
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


try:
    model = load_model()
    collection = load_collection()

    st.success(
        f"Connected to ChromaDB | "
        f"Stored chunks: {collection.count()}"
    )

except Exception as e:
    st.error(f"Could not initialize the system: {e}")
    st.stop()


# -----------------------------
# Search Interface
# -----------------------------

st.subheader("Semantic Search")

query = st.text_input(
    "Enter your question",
    placeholder="How do I create an agent in LangChain?"
)

top_k = st.slider(
    "Number of results to retrieve",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Search", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documentation..."):

            # Convert query into embedding
            query_embedding = model.encode(
                query,
                convert_to_numpy=True
            ).tolist()

            # Search ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        st.subheader(
            f"Retrieved {len(documents)} results"
        )

        for i, (doc, metadata, distance) in enumerate(
            zip(documents, metadatas, distances)
        ):

            # Collection uses cosine distance
            similarity = 1 - distance

            with st.expander(
                f"Result {i+1} | Similarity: {similarity:.4f}",
                expanded=(i == 0)
            ):

                st.markdown("**Document content**")
                st.write(doc)

                st.markdown("**Metadata**")

                st.write({
                    "Source": metadata.get("source"),
                    "Chunk Index": metadata.get("chunk_index")
                })

                st.caption(
                    f"Cosine distance: {distance:.4f}"
                )
