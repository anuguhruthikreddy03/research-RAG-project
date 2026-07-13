# src/generation/chain.py
import os
from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Client-side rate limiter for Gemini Free Tier (max 15 RPM)
# 1 request every 5 seconds = max 12 RPM
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)

load_dotenv()

def get_reranked_context(query_text: str, vectorstore, top_k: int = 5) -> str:
    """
    Executes raw database vector similarity queries bypassing deprecated retrieval 
    wrappers and computes high-accuracy contextual reranking using Jina AI's Cross-Encoder.
    """
    if not vectorstore:
        return "Please upload a document to proceed."
        
    # Query double the candidate chunk arrays to offer appropriate variance for the reranker
    raw_documents = vectorstore.similarity_search(query_text, k=top_k * 2)
    
    if not raw_documents:
        return "I could not locate any context matching this parameter in the document."
        
    description_hits = [doc.page_content for doc in raw_documents]
    
    # Configure the cross-encoder using a localized caching space to verify write stability
    cache_dir = os.path.join(os.getcwd(), ".fastembed_cache")
    reranker = TextCrossEncoder(
        model_name='jinaai/jina-reranker-v2-base-multilingual',
        cache_dir=cache_dir
    )
    
    new_scores = list(reranker.rerank(query_text, description_hits))
    
    # Sort indexes cleanly using Python list lambda sorting
    ranking = [(i, score) for i, score in enumerate(new_scores)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    top_rankings = ranking[:top_k]
    return "\n\n".join(description_hits[rank[0]] for rank in top_rankings)


def build_dynamic_rag_chain(vectorstore):
    """
    Constructs an updated functional LCEL chain routing state streams explicitly 
    via RunnableLambdas.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        rate_limiter=rate_limiter,
        max_retries=6
    )
    parser = StrOutputParser()
    
    system_prompt = (
        "You are an expert academic research assistant.\n"
        "Answer the user's question using only the following retrieved context.\n"
        "If you do not know the answer, say: 'I could not find this in the document.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    # Modern explicit functional mapping topology
    rag_chain = (
        {
            "context": RunnableLambda(lambda inputs: get_reranked_context(inputs["question"], vectorstore)),
            "question": lambda inputs: inputs["question"]
        }
        | prompt 
        | llm 
        | parser
    )
    
    return rag_chain