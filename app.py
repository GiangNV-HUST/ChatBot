import streamlit as st
import logging
import traceback
from rag_chatbot import RAGChatbotSystem

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_streamlit_ui():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 RAG Chatbot với LangChain và FAISS Vector Database")

    with st.spinner("Đang kiểm tra hệ thống..."):
        if "chatbot" not in st.session_state:
            st.write("Khởi tạo chatbot...")
            try:
                st.session_state.chatbot = RAGChatbotSystem()
                st.success("Chatbot đã sẵn sàng!")
            except Exception as e:
                st.error(f"Lỗi khởi tạo chatbot: {str(e)}")
                return

    with st.sidebar:
        st.header("⚙️ Cấu hình")

        # Tải lên tài liệu
        st.subheader("Tải lên tài liệu")
        uploaded_files = st.file_uploader(
            "Chọn file để tải lên:",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        )

        save_option = st.checkbox("Lưu file vào thư mục dữ liệu", value=True)

        if st.button("Xử lý file đã tải lên"):
            if uploaded_files:
                total_success = 0
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Đang xử lý {uploaded_file.name}..."):
                        success, message = (
                            st.session_state.chatbot.add_uploaded_document(
                                uploaded_file, save_to_data_dir=save_option
                            )
                        )
                        if success:
                            st.success(message)
                            total_success += 1
                        else:
                            st.error(message)

                if total_success > 0:
                    st.success(
                        f"Đã xử lý thành công {total_success}/{len(uploaded_files)} file"
                    )
            else:
                st.warning("Vui lòng tải lên ít nhất một file")

        # Thêm tài liệu từ web
        st.subheader("Thêm tài liệu từ web")
        url = st.text_input("Nhập URL:", key="url_input")
        if st.button("Thêm URL"):
            if url:
                with st.spinner(f"Đang tải dữ liệu từ {url}..."):
                    success, message = st.session_state.chatbot.add_web_document(url)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Vui lòng nhập URL")

        # Quản lý hội thoại
        st.subheader("Quản lý hội thoại")
        if st.button("Xóa lịch sử hội thoại"):
            st.session_state.chatbot.clear_memory()
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Hỏi tôi bất cứ điều gì."}
            ]
            st.success("Đã xóa lịch sử hội thoại")

        # Quản lý dữ liệu
        st.subheader("Quản lý dữ liệu")
        if st.button("Tải lại dữ liệu"):
            with st.spinner("Đang tải lại dữ liệu..."):
                st.session_state.chatbot._setup_system()
                st.session_state.chatbot._setup_chain()
            st.success("Đã tải lại dữ liệu thành công")

        # Thông tin hệ thống
        st.subheader("Thông tin hệ thống")
        st.info(
            """
            **Mô hình sử dụng:**
            - Embedding: all-MiniLM-L6-v2
            - LLM: Vinallama-7B-Chat
            - Vector DB: FAISS
            
            **Định dạng hỗ trợ:**
            - PDF (.pdf)
            - Word (.docx)
            - Text (.txt)
            - CSV (.csv)
            - Excel (.xlsx, .xls)
            """
        )

    # Khởi tạo lịch sử hội thoại
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Hỏi tôi bất cứ điều gì."}
        ]

    # Hiển thị lịch sử hội thoại
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Nguồn:**")
                for source in message["sources"]:
                    st.markdown(f"- {source}")

    # Xử lý input từ người dùng
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Thêm tin nhắn của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Xử lý và hiển thị phản hồi
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                response = st.session_state.chatbot.chat(prompt)
                st.markdown(response["answer"])
                if response["sources"]:
                    st.markdown("**Nguồn:**")
                    for source in response["sources"]:
                        st.markdown(f"- {source}")
                
                # Thêm phản hồi vào lịch sử
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    }
                )


def main():
    try:
        setup_streamlit_ui()
    except Exception as e:
        st.error(f"Lỗi khi chạy ứng dụng: {str(e)}")
        st.error(traceback.format_exc())
        logger.error(f"Lỗi ứng dụng: {e}")


if __name__ == "__main__":
    main()