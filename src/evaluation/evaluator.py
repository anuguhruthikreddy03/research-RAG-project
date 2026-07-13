import os
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from datasets import Dataset

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langfuse.langchain import CallbackHandler

# Client-side rate limiter for Gemini Free Tier (max 15 RPM)
# 1 request every 5 seconds = max 12 RPM
eval_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)

class EvaluationSample(BaseModel):
    question: str = Field(description="A clear and specific question derived from the context.")
    ground_truth: str = Field(description="The correct, detailed answer to the question based ONLY on the context.")

class EvaluationSet(BaseModel):
    samples: List[EvaluationSample]

def run_dynamic_evaluation(processed_documents, rag_chain, vectorstore, num_questions: int = 3):
    """
    Synthesizes test sets and evaluates RAG pipeline metrics natively 
    using modern Class structures, SingleTurnSamples, wrappers, and Langfuse tracing.
    """
    if not processed_documents or not rag_chain or not vectorstore:
        return pd.DataFrame({"Error": ["Missing engine context prerequisites."]})

    google_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        rate_limiter=eval_rate_limiter,
        max_retries=6
    )
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    evaluator_llm = LangchainLLMWrapper(google_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(google_embeddings)

    # Select representative chunks evenly across the document
    step = max(1, len(processed_documents) // num_questions)
    selected_docs = []
    for i in range(num_questions):
        idx = min(i * step, len(processed_documents) - 1)
        selected_docs.append(processed_documents[idx])
    
    # Ensure we limit to exactly num_questions even if list is short
    selected_docs = selected_docs[:num_questions]
    
    context_text = "\n\n".join(
        f"--- Chunk {i+1} ---\n{doc.page_content}" 
        for i, doc in enumerate(selected_docs)
    )
    prompt_text = (
        f"You are a system designed to generate evaluation datasets for RAG pipelines.\n"
        f"Based ONLY on the provided text chunks, generate exactly {num_questions} diverse and challenging questions "
        f"along with their detailed correct ground truth answers.\n\n"
        f"Generate exactly one question-answer pair per chunk so that the questions cover different parts of the context.\n\n"
        f"Context Chunks:\n{context_text}"
    )

    print(f"Ragas is generating {num_questions} questions via custom structured generator...")
    structured_llm = google_llm.with_structured_output(EvaluationSet)
    response = structured_llm.invoke(prompt_text)
    
    generated_questions = [sample.question for sample in response.samples]
    ground_truths = [sample.ground_truth for sample in response.samples]
    
    # Initialize Langfuse Callback handler early if credentials exist
    langfuse_callback = None
    if os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY'):
        print("Langfuse monitoring detected. Initializing cloud observability tracing handler...")
        langfuse_callback = CallbackHandler()

    sample_list = []

    from src.generation.chain import get_reranked_context
    for question, ground_truth in zip(generated_questions, ground_truths):
        try:
            config = {}
            if langfuse_callback:
                config["callbacks"] = [langfuse_callback]
            res_answer = rag_chain.invoke({"question": question}, config=config)
        except Exception as err:
            res_answer = f"Pipeline Inference Error: {err}"
            
        # Retrieve context matching the parameters
        retrieved_text = get_reranked_context(question, vectorstore, top_k=5)
        
        sample = SingleTurnSample(
            user_input=question,
            reference=ground_truth,
            response=res_answer,
            retrieved_contexts=[retrieved_text]
        )
        sample_list.append(sample)

    evaluation_dataset = EvaluationDataset(samples=sample_list)

    f_metric = Faithfulness(llm=evaluator_llm)
    ar_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

    metrics_list = [f_metric, ar_metric]

    # Pass LLM and embeddings directly to evaluate (new syntax)
    print("Computing metrics via Native Ragas Engine...")
    
    # Control concurrency using RunConfig to avoid 429 rate limit
    run_config = RunConfig(max_workers=1, timeout=300)
    
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics_list,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        callbacks=[langfuse_callback] if langfuse_callback else [],
        run_config=run_config
    )
    
    return result.to_pandas()