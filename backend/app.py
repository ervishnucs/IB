import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_cohere import ChatCohere
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from any origin (for development purposes only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
#os.environ["COHERE_API_KEY"] = "0rFtyFbKgoReHD5oMnk3jlyGYTOy4lA5qdDw8Efy"

# Initialize LLM and embeddings
llm = ChatCohere(model="command-r-plus")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load insurance policy documents
#loader = PyPDFDirectoryLoader(r"C:\Users\pmlba\OneDrive\Desktop\policies")
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize Chroma Vector Store
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
retriever = vectorstore.as_retriever(search_type="mmr")

# Create prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "reformulate it if needed and answer the question concisely."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("human", "{input}"),
    ]
)

system_prompt = (
        "You are an AI India's FirstInsurance Bot designed to answer questions regarding insurance policy documents."
    "Your responses must be accurate, concise, and based only on the provided policy data. "
    "If User says any greetings make sure your concerstion is polite."
    "Always provide monetary values in Indian Rupees (₹) and ensure that answers are well-structured and free from hallucinations."
    "Guidelines for responses:"
    "1. Always refer to the specific insurance policies provided for all answers ."
    "2. When asked for policy suggestions, only suggest from the provided documents."
    "3. For any calculation-related queries, provide basic calculations or approximate answers based on available information."
    "4. If the user asks irrelevant or nonsensical questions, kindly inform them that you are an AI Insurance Bot "
    "and can only assist with policy-related queries."
    "5. Maintain a professional tone and ensure responses are concise and to the point,"
    "especially when extracting quantitative details like premiums, coverage amounts, and policy terms." 
    "6.After providing an answer, ask if the user needs further clarification or suggestions. "
    "Provide the answer within 50 words"

   "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store chat history
store = {}

class QuestionRequest(BaseModel):
    question: str
    session_id: str

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define the conversational RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
import logging

# Add logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Define the FastAPI route
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Invoke the chain with session history and question
        session_history = get_session_history(request.session_id).messages
        response = conversational_rag_chain.invoke(
            {"input": request.question, "chat_history": session_history},
            {"configurable": {"session_id": request.session_id}}
        )
        if "answer" in response:
                 return JSONResponse(content={"response": response["answer"]})
        else:
                raise HTTPException(status_code=500, detail="Failed to retrieve an answer from the chain")

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
