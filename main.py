from dotenv import load_dotenv
import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from time import sleep
from product import products
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tiktoken
import logging
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()
# logger = logging.getLogger("ApplicationLog")
logging.basicConfig(
    filename='application.log',
    level=logging.DEBUG
)

st.set_page_config(
    page_title="対話型商品レコメンド生成AIアプリ"
)

st.markdown('## 対話型商品レコメンド生成AIアプリ')

with st.chat_message("assistant", avatar="images/f_f_object_174_s512_f_object_174_2bg.png"):
    # st.markdown("こちらは対話型で商品レコメンドを行う生成AIチャットボットです。条件検索とチャットを活用し「こんな商品が欲しい」といった情報を入力すれば、生成AIがあなたに合った商品をレコメンドします。")
    st.info("私（生成AI）があなたに合った商品をレコメンドします！「こんな商品が欲しい」といった情報・要望をチャット欄からぜひご入力ください！")

if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.chat_history = []

    # データベースとChainsの作成

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="images/f_f_object_153_s512_f_object_153_2bg.png"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="images/f_f_object_174_s512_f_object_174_2bg.png"):
            product = message["content"]
            try:
                name = product['name'] # わざと例外を発生させるために暫定対応
                st.markdown(product["message"])
                st.success(f"""{product['name']}\n
                        価格：{product['price']}""")
                st.image(f"images/{product['image_file_name']}")
                st.code(f"""
                {product['description']}
                """, language=None, wrap_lines=True)
                st.link_button("商品ページを開く", type="primary", use_container_width=True, url="https://google.com")
            except:
                st.markdown(product["message"])

input_message = st.chat_input("例： 防水機能のあるカメラ")

if input_message:
    st.session_state.messages.append({"role": "user", "content": input_message})
    with st.chat_message("user", avatar="images/f_f_object_153_s512_f_object_153_2bg.png"):
        st.markdown(input_message)

    res_box = st.empty()
    with st.spinner('レコメンドする商品の検討中...'):
        with st.chat_message("assistant", avatar="images/f_f_object_174_s512_f_object_174_2bg.png"):
            loader = CSVLoader(file_path="data/products.csv")
            docs = loader.load()

            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, streaming=True)
            db = Chroma.from_documents(docs, embedding=embeddings)
            # if os.path.isdir(".db"):
            #     db = Chroma(persist_directory=".db", embedding_function=embeddings)
            # else:
            #     db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=".db")
            retriever = db.as_retriever(search_kwargs={"k": 5})
            result = retriever.get_relevant_documents(input_message)
            print(f"====================={result}")

            question_generator_template = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"
            question_generator_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_generator_template),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_template = """
            あなたは、ユーザー入力に対しておすすめの商品をレコメンドするAIです。
            以下の各項目を、以下のJSON形式のフォーマットで返却してください。

            【返却するJSON形式の文字列のフォーマットの一部】
            {'name': '商品名', 'category': 'カテゴリ', 'price': '価格', 'maker': 'メーカー', 'recommended_people': 'おすすめな人', 'review_number': '口コミの数', 'score': '口コミの評価平均点', 'file_name': 'ファイル名', 'description': '商品説明'}

            【項目】
            ・name
            ・category
            ・price
            ・maker
            ・recommended_people
            ・review_number
            ・score
            ・file_name
            ・description

            {context}
            """
            question_answer_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_answer_template),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, question_generator_prompt
            )
            question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # if input_message == "料理が楽になる包丁を教えて":
            #     product = products[0]
            # elif input_message == "コンセントがないところでも充電できる商品":
            #     product = products[1]
            # elif input_message == "冷たい状態を長時間キープするボトル":
            #     product = products[2]
            # elif input_message == "おすすめのコーヒー豆を教えてください。":
            #     product = products[3]
            # elif input_message == "通気性が良い、ペット用のバッグ":
            #     product = products[4]
            # elif input_message == "ペットの寝具でおすすめのもの":
            #     product = products[5]
            # elif input_message == "初心者におすすめのキャンプ用品":
            #     product = products[6]
            # elif input_message == "キャンプで、他におすすめの商品はある？":
            #     product = products[7]
            # else:
            #     product = ""
            
            # if product:

            # 商品情報を取得
            # result = history_aware_retriever.get_relevant_documents(input_message)
            result = rag_chain.invoke({"input": input_message, "chat_history": st.session_state.chat_history})
            print(f"********************{result}")
            st.session_state.chat_history.extend([HumanMessage(content=input_message), AIMessage(content=result[0])])

            # レスポンスを辞書に変換
            product_lines = result['context'][0].page_content.split("\n")
            product = {item.split(": ")[0]: item.split(": ")[1] for item in product}
            product["message"] = "以下の商品をご提案いたします。"

            # 商品の詳細情報を表示
            st.markdown(product["message"])
            st.success(f"""{product['name']}\n
                    価格：{product['price']}""")
            st.image(f"images/products/{product['file_name']}")
            st.code(f"""
            {product['description']}
            """, language=None, wrap_lines=True)
            st.link_button("商品ページを開く", type="primary", use_container_width=True, url="https://google.com")
            # else:
            #     if input_message == "切れるもの":
            #         message = "もう少し具体的にお聞かせください。"
            #     else:
            #         message = "レコメンド対象の商品が見つかりませんでした。別の入力内容で検索してください。"
            #     st.markdown(message)
            #     product = {"message": message}
            
            st.session_state.messages.append({"role": "assistant", "content": product})
    st.rerun()