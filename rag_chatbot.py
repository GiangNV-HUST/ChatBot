import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_core.documents import Document
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class RAGChatbotSystem:
    def __init__(self, data_dir="data", persist_dir="faiss_index"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        logger.info(
            f"Khởi tạo RAG Chatbot với thư mục dữ liệu: {os.path.abspath(data_dir)}"
        )

        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.error(f"Lỗi khi tải embedding model: {e}")
            raise Exception(f"Không thể tải embedding model: {str(e)}")

        model_path = os.path.join(os.getcwd(), "models", "vinallama-7b-chat_q5_0.gguf")
        if not os.path.exists(model_path):
            logger.error(f"Model không tồn tại tại: {model_path}")
            raise FileNotFoundError(f"Model không tồn tại tại: {model_path}")

        try:
            self.llm = CTransformers(
                model=model_path,
                model_type="llama",
                config={
                    "max_new_tokens": 256,
                    "temperature": 0.2,
                    "context_length": 1024,
                },
            )
        except Exception as e:
            logger.error(f"Không thể tải model LLM: {e}")
            from langchain.llms import FakeListLLM

            self.llm = FakeListLLM(responses=["Model không khả dụng."])

        self.memory = InMemoryChatMessageHistory()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self._ensure_data_directory()

        if os.path.exists(f"{persist_dir}/index.faiss") and os.path.exists(
            f"{persist_dir}/index.pkl"
        ):
            logger.info(f"Tìm thấy FAISS index tại {persist_dir}, đang tải...")
            try:
                self.vectordb = FAISS.load_local(
                    persist_dir,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.error(f"Lỗi khi tải FAISS index: {e}")
                self._setup_system()
        else:
            logger.info("Không tìm thấy vector database, đang tạo mới...")
            self._setup_system()

        self._setup_chain()

    def _ensure_data_directory(self):
        data_path = Path(self.data_dir)
        data_path.mkdir(exist_ok=True)
        files = list(data_path.glob("**/*.txt"))
        if not files:
            logger.info(
                f"Không tìm thấy file trong {self.data_dir}, đang tạo file mẫu..."
            )
            sample_file = data_path / "sample.txt"
            try:
                with open(sample_file, "w", encoding="utf-8") as f:
                    f.write("Dữ liệu mẫu cho RAG chatbot.")
                logger.info(f"Đã tạo file mẫu tại {sample_file}")
            except Exception as e:
                logger.error(f"Không thể tạo file mẫu: {e}")

    def _setup_system(self):
        documents = self._load_documents()
        if not documents:
            documents = [
                Document(page_content="Tài liệu mẫu.", metadata={"source": "sample"})
            ]
        splits = self.text_splitter.split_documents(documents)
        try:
            self.vectordb = FAISS.from_documents(splits, self.embedding_model)
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vectordb.save_local(self.persist_dir)
        except Exception as e:
            logger.error(f"Lỗi khi tạo FAISS index: {e}")
            raise

    def _load_documents(self):
        documents = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            return documents
        try:
            # Tải tài liệu văn bản
            txt_loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.txt",
                loader_cls=lambda x: TextLoader(x, encoding="utf-8"),
            )
            documents.extend(txt_loader.load())

            # Tải tài liệu PDF
            pdf_loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
            )
            documents.extend(pdf_loader.load())

            # Tải tài liệu Word
            docx_loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
            )
            documents.extend(docx_loader.load())

            # Tải tài liệu CSV
            csv_loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.csv",
                loader_cls=lambda x: CSVLoader(x, encoding="utf-8"),
            )
            documents.extend(csv_loader.load())

            # Tải tài liệu Excel
            excel_loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.xlsx",
                loader_cls=UnstructuredExcelLoader,
            )
            documents.extend(excel_loader.load())

        except Exception as e:
            logger.error(f"Lỗi khi tải tài liệu: {e}")

        logger.info(f"Đã tải tổng cộng {len(documents)} tài liệu")
        return documents

    def add_web_document(self, url):
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)
            new_db = FAISS.from_documents(splits, self.embedding_model)
            if hasattr(self, "vectordb"):
                self.vectordb.merge_from(new_db)
            else:
                self.vectordb = new_db
            self.vectordb.save_local(self.persist_dir)
            return True, f"Đã thêm {len(splits)} phân đoạn từ {url}"
        except Exception as e:
            logger.error(f"Lỗi khi thêm tài liệu từ web: {e}")
            return False, f"Lỗi: {str(e)}"

    def add_uploaded_document(self, uploaded_file, save_to_data_dir=True):
        try:
            # Xác định loại file và loader tương ứng
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            # Lưu file tạm thời để xử lý
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Chọn loader phù hợp với loại file
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(tmp_path)
            elif file_extension == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif file_extension == ".csv":
                loader = CSVLoader(tmp_path, encoding="utf-8")
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(tmp_path)
            else:
                os.unlink(tmp_path)
                return False, f"Định dạng file {file_extension} không được hỗ trợ"

            # Tải và xử lý tài liệu
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)

            # Cập nhật vector database
            new_db = FAISS.from_documents(splits, self.embedding_model)
            if hasattr(self, "vectordb"):
                self.vectordb.merge_from(new_db)
            else:
                self.vectordb = new_db
            self.vectordb.save_local(self.persist_dir)

            # Lưu file vào thư mục data nếu được yêu cầu
            if save_to_data_dir:
                os.makedirs(self.data_dir, exist_ok=True)
                dest_path = os.path.join(self.data_dir, uploaded_file.name)
                shutil.copy(tmp_path, dest_path)
                logger.info(f"Đã lưu file {uploaded_file.name} vào thư mục dữ liệu")

            # Xóa file tạm
            os.unlink(tmp_path)

            return True, f"Đã thêm {len(splits)} phân đoạn từ {uploaded_file.name}"
        except Exception as e:
            logger.error(f"Lỗi khi thêm tài liệu từ upload: {e}")
            return False, f"Lỗi: {str(e)}"

    def _setup_chain(self):
        retriever = self.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
        )
        template = """
        Context: {context}
        Lịch sử hội thoại: {chat_history}
        Câu hỏi: {question}
        Trả lời:
        """
        QA_PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"], template=template
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
        )
        self.chain = RunnableWithMessageHistory(
            chain,
            lambda: self.memory,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def chat(self, query):
        try:
            result = self.chain.invoke(
                {"question": query}, config={"configurable": {"session_id": "default"}}
            )
            sources = [
                doc.metadata.get("source", "Unknown")
                for doc in result["source_documents"]
            ]
            return {"answer": result["answer"], "sources": list(dict.fromkeys(sources))}
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {e}")
            return {"answer": f"Lỗi: {str(e)}", "sources": []}

    def clear_memory(self):
        self.memory.clear()
        logger.info("Đã xóa lịch sử hội thoại")