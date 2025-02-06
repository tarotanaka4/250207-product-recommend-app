import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import HumanMessage, AIMessage
import functions as ft
import constants as ct

# 各種設定
load_dotenv()
st.set_page_config(
    page_title=ct.APP_NAME
)

# 初期表示
st.markdown(f"## {ct.APP_NAME}")
with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
    st.markdown("こちらは対話型の商品レコメンド生成AIアプリです。「こんな商品が欲しい」という情報・要望を画面下部のチャット欄から送信いただければ、おすすめの商品をレコメンドいたします。")
    st.markdown("**入力例**")
    st.info("""
    - 「折り畳み傘」
    - 「インテリアとしても使える芳香剤」
    - 「長時間使える、高音質なワイヤレスイヤホン」
    """)

# 初期処理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 再描画時にこれまでのメッセージ一覧を表示
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
            product = message["content"]
            ft.display_product(product)

# ユーザー入力の受け付け
input_message = st.chat_input("例： 防水機能のあるカメラ")

if input_message:
    # ユーザーメッセージを表示
    with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
        st.markdown(input_message)

    res_box = st.empty()
    with st.spinner('レコメンドする商品の検討中...'):
        # レスポンスメッセージの取得・表示
        with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
            retriever = ft.create_retriever()
            result = retriever.invoke(input_message)

            # レスポンスのテキストを辞書に変換
            product_lines = result[0].page_content.split("\n")
            product = {item.split(": ")[0]: item.split(": ")[1] for item in product_lines}

            # 商品の詳細情報を表示
            ft.display_product(product)
            
    # メッセージ一覧に追加
    st.session_state.messages.append({"role": "user", "content": input_message})
    st.session_state.messages.append({"role": "assistant", "content": product})