import os

from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)

LLM_BASE_URL = "https://mkp-api.fptcloud.com"
LLM_API_KEY = os.environ["FPT_API_KEY"]
EMB_BASE_URL = "http://localhost:8080/v1"  # since using TEI locally

judge_llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model="DeepSeek-V3",
    temperature=0.0,
)

emb = OpenAIEmbeddings(
    base_url=EMB_BASE_URL,  # TEI's OpenAI-compatible route
    api_key="not-needed",  # TEI often runs without auth; LangChain still requires a string
    model="",  # must match the model you launched in TEI
)

jsonl_path = "eval jsonl data path"

eval_dataset = load_dataset("jsonl", data_files=jsonl_path)

result = evaluate(
    eval_dataset,
    llm=judge_llm,
    embeddings=emb,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
    ],
    column_map={"question": "question", "contexts": "contexts", "answer": "answer"},
)

result_df = result.to_pandas()
result_df.to_csv("result.csv", index=False)
