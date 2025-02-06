from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from typing import List
from sudachipy import tokenizer, dictionary
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def preprocess_func(text):
    """
    形態素解析による日本語の単語分割
    Args:
        text: 単語分割対象のテキスト
    """
    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))
    return words

def create_retriever():
    """
    Retriever作成
    """
    loader = CSVLoader(file_path="data/products.csv")
    docs = loader.load()

    docs_contents = []
    for doc in docs:
        docs_contents.append(doc.page_content)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embedding=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_texts(
        docs_contents,
        preprocess_func=preprocess_func,
        k=3
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever

def display_product(product):
    """
    商品情報の表示
    Args:
        product: 商品オブジェクト
    """
    st.markdown("以下の商品をご提案いたします。")
    st.success(f"""
            商品名：{product['name']}（商品ID: {product['id']}）\n
            価格：{product['price']}
    """)
    st.code(f"""
        商品カテゴリ：{product['category']}\n
        メーカー：{product['maker']}\n
        評価：{product['score']}({product['review_number']}件)
    """, language=None, wrap_lines=True)
    st.image(f"images/products/{product['file_name']}", width=400)
    st.code(f"""
        {product['description']}
    """, language=None, wrap_lines=True)
    st.markdown("**こんな方におすすめ！**")
    st.info(product["recommended_people"])
    st.link_button("商品ページを開く", type="primary", use_container_width=True, url="https://google.com")