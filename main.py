from dotenv import load_dotenv
import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from time import sleep
from product import products

load_dotenv()

st.set_page_config(
    page_title="対話型商品レコメンド生成AIアプリ"
)

st.markdown('## 対話型商品レコメンド生成AIアプリ')

with st.chat_message("assistant", avatar="images/f_f_object_174_s512_f_object_174_2bg.png"):
    # st.markdown("こちらは対話型で商品レコメンドを行う生成AIチャットボットです。条件検索とチャットを活用し「こんな商品が欲しい」といった情報を入力すれば、生成AIがあなたに合った商品をレコメンドします。")
    st.markdown("私（生成AI）があなたに合った商品をレコメンドします！「こんな商品が欲しい」といった情報・要望をチャット欄からぜひご入力ください！")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                st.info(f"""{product['name']}\n
                        価格：{product['price']}""")
                st.image(f"images/{product['image_file_name']}")
                st.code(f"""
                {product['description']}
                """, language=None, wrap_lines=True)
                st.link_button("商品ページを開く", type="primary", use_container_width=True, url="https://google.com")
            except:
                st.markdown(product["message"])

chat_message = st.chat_input("例： 防水機能のあるカメラ")

if chat_message:
    st.session_state.messages.append({"role": "user", "content": chat_message})
    with st.chat_message("user", avatar="images/f_f_object_153_s512_f_object_153_2bg.png"):
        st.markdown(chat_message)

    with st.spinner('レコメンド商品の検討中...'):
        sleep(2)
        with st.chat_message("assistant", avatar="images/f_f_object_174_s512_f_object_174_2bg.png"):
            if chat_message == "料理が楽になる包丁を教えて":
                product = products[0]
            elif chat_message == "コンセントがないところでも充電できる商品":
                product = products[1]
            elif chat_message == "冷たい状態を長時間キープするボトル":
                product = products[2]
            elif chat_message == "おすすめのコーヒー豆を教えてください。":
                product = products[3]
            elif chat_message == "通気性が良い、ペット用のバッグ":
                product = products[4]
            elif chat_message == "ペットの寝具でおすすめのもの":
                product = products[5]
            elif chat_message == "初心者におすすめのキャンプ用品":
                product = products[6]
            elif chat_message == "キャンプで、他におすすめの商品はある？":
                product = products[7]
            else:
                product = ""
            
            if product:
                product["message"] = "以下の商品をご提案いたします。"
                st.markdown(product["message"])
                st.info(f"""{product['name']}\n
                        価格：{product['price']}""")
                st.image(f"images/{product['image_file_name']}")
                st.code(f"""
                {product['description']}
                """, language=None, wrap_lines=True)
                st.link_button("商品ページを開く", type="primary", use_container_width=True, url="https://google.com")
            else:
                if chat_message == "切れるもの":
                    message = "もう少し具体的にお聞かせください。"
                else:
                    message = "レコメンド対象の商品が見つかりませんでした。別の入力内容で検索してください。"
                st.markdown(message)
                product = {"message": message}
            
            st.session_state.messages.append({"role": "assistant", "content": product})
    st.rerun()