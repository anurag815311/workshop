
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai


# -------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="My AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 My AI Study Assistant")
st.write(
    "Ask questions about DBMS, COA, and campus policies."
)


# -------------------------------
# 2. API KEY CONFIGURATION
# -------------------------------

st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Enter your Gemini API key",
    type="password",
    help="Your API key is used to generate answers with Gemini."
)


# -------------------------------
# 3. KNOWLEDGE BASE
# -------------------------------

documents = [
    "DBMS: ACID stands for Atomicity, Consistency, Isolation, and Durability. "
    "These properties help ensure reliable database transactions.",

    "DBMS: A vector database stores high-dimensional vector embeddings "
    "and supports semantic search.",

    "COA: Cache memory stores frequently accessed data to reduce "
    "average memory access time.",

    "COA: Amdahl's law estimates the theoretical speedup of a system "
    "when only part of the system is improved.",

    "Campus Policy: Students must carry their institutional ID cards "
    "while on campus.",

    "Campus Policy: Students should consult the official campus portal "
    "for the latest attendance and leave policies."
]


# -------------------------------
# 4. LOAD THE EMBEDDING MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# 5. PREPARE DOCUMENT EMBEDDINGS
# -------------------------------

@st.cache_resource
def load_knowledge_base():
    model = load_model()

    embeddings = model.encode(
        documents,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return embeddings


# -------------------------------
# 6. RETRIEVE RELEVANT DOCUMENTS
# -------------------------------

def retrieve_documents(question):
    model = load_model()
    document_embeddings = load_knowledge_base()

    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = model.similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort(
        descending=True
    )[:2].tolist()

    retrieved_documents = [
        documents[index] for index in ranked_indices
    ]

    scores = [
        float(similarities[index])
        for index in ranked_indices
    ]

    return retrieved_documents, scores


# -------------------------------
# 7. GENERATE ANSWER USING GEMINI
# -------------------------------

def generate_answer(question, api_key, context):
    client = genai.Client(api_key=api_key)

    system_instruction = """
    You are a helpful academic study assistant.

    Answer the user's question using only the supplied context.
    Explain concepts clearly and in student-friendly language.

    If the context does not contain enough information to answer,
    explicitly say that the information is unavailable in the
    provided documents.

    Do not invent campus policies, facts, or unsupported details.
    """

    prompt = f"""
    Retrieved document context:

    {chr(10).join(context)}

    Student's question:
    {question}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.2
        }
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response. Please try again."
        )

    return response.text


# -------------------------------
# 8. USER INTERACTION
# -------------------------------

st.subheader("Ask a Question")

question = st.text_input(
    "Your question",
    placeholder="What are ACID properties?",
    key="question"
)

ask_button = st.button(
    "Ask AI",
    type="primary",
    use_container_width=True
)


# -------------------------------
# 9. PROCESS THE QUESTION
# -------------------------------

if ask_button:

    if not api_key.strip():
        st.warning(
            "Please enter your Gemini API key in the sidebar."
        )

    elif not question.strip():
        st.warning(
            "Please enter a question before clicking Ask AI."
        )

    else:
        try:
            with st.spinner(
                "Searching documents and generating your answer..."
            ):
                retrieved_documents, similarity_scores = (
                    retrieve_documents(question.strip())
                )

                answer = generate_answer(
                    question.strip(),
                    api_key.strip(),
                    retrieved_documents
                )

            st.subheader("Answer")
            st.markdown(answer)

            with st.expander(
                "View Retrieved Documents"
            ):
                st.caption(
                    "These are the two documents selected "
                    "for answering your question."
                )

                for index, document in enumerate(
                    retrieved_documents
                ):
                    st.markdown(
                        f"**Document {index + 1}**"
                    )
                    st.write(document)

                    st.caption(
                        f"Cosine similarity: "
                        f"{similarity_scores[index]:.3f}"
                    )

        except Exception as error:
            st.error(
                "The application could not complete your request."
            )

            st.caption(
                "Check your API key, internet connection, "
                "model availability, and API quota."
            )

            with st.expander("Technical error details"):
                st.code(str(error))


# -------------------------------
# 10. FOOTER
# -------------------------------

st.divider()

st.caption(
    "My AI Study Assistant | Built with Python, Streamlit, "
    "Sentence Transformers, and Gemini"
)
