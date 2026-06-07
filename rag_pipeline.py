"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation

RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp:
- Retrieval (Tìm kiếm): Tìm các đoạn văn bản liên quan từ knowledge base
- Generation (Sinh văn bản): Dùng LLM để tạo câu trả lời dựa trên ngữ cảnh tìm được
"""

# Import các module embedding và prompt từ LangChain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# RunnablePassthrough: truyền dữ liệu đầu vào qua mà không thay đổi
# RunnableParallel: chạy nhiều chain song song
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# StrOutputParser: chuyển output của LLM thành chuỗi string thuần
from langchain_core.output_parsers import StrOutputParser

# Import các model LLM và embedding của OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# init_chat_model: hàm tiện ích để khởi tạo chat model theo tên
from langchain.chat_models import init_chat_model

# Chroma: vector store để lưu và tìm kiếm embeddings
from langchain_chroma import Chroma

# Document: class đại diện cho một tài liệu (nội dung + metadata)
from langchain_core.documents import Document

# RecursiveCharacterTextSplitter: tách văn bản dài thành các chunk nhỏ hơn
from langchain_text_splitters import RecursiveCharacterTextSplitter

# BaseModel, Field: dùng để định nghĩa cấu trúc output có schema rõ ràng (Pydantic)
from pydantic import BaseModel, Field
from typing import List

# load_dotenv: đọc biến môi trường từ file .env (API keys, v.v.)
from dotenv import load_dotenv
import tempfile  # tạo thư mục/file tạm thời trên hệ thống

# Tải các biến môi trường từ file .env (ví dụ: OPENAI_API_KEY)
load_dotenv()

# Khởi tạo model embedding để chuyển văn bản thành vector số học
# "text-embedding-3-small" là model nhỏ gọn, tiết kiệm chi phí của OpenAI
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# Knowledge Base mẫu — đây là "tài liệu" mà RAG sẽ tìm kiếm
# Trong thực tế, đây có thể là PDF, database, website, v.v.
# ============================================================
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

# Khởi tạo LLM (Large Language Model) — model sẽ tạo ra câu trả lời
# temperature=0.2: độ ngẫu nhiên thấp → câu trả lời ổn định, ít "sáng tạo" hơn
llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)


def create_kb():
    """Tạo vector store từ knowledge base văn bản."""

    # Bước 1: Tách knowledge base thành các chunk nhỏ
    # chunk_size=500: mỗi chunk tối đa 500 ký tự
    # chunk_overlap=50: 50 ký tự cuối của chunk trước sẽ lặp lại ở chunk sau
    #   → giúp không mất ngữ cảnh ở ranh giới giữa các chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Gói văn bản vào đối tượng Document kèm metadata (nguồn gốc)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    # Thực hiện tách văn bản → danh sách các Document nhỏ hơn
    chunks = splitter.split_documents([doc])

    # Bước 2: Tạo vector store từ các chunk
    # Chroma.from_documents sẽ:
    #   - Gọi embeddings_model để chuyển mỗi chunk thành vector
    #   - Lưu các vector vào Chroma (vector database)
    # persist_directory: thư mục lưu dữ liệu (dùng thư mục tạm để demo)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


def demo_basic_rag():
    """Demo RAG cơ bản: truy vấn → tìm ngữ cảnh → sinh câu trả lời."""

    # Tạo vector store từ knowledge base
    vector_store = create_kb()

    # Tạo retriever từ vector store
    # search_type="similarity": tìm kiếm theo độ tương đồng cosine
    # k=2: lấy 2 chunk liên quan nhất
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # Định nghĩa prompt template cho RAG
    # {context}: sẽ được điền bởi các đoạn văn bản tìm được
    # {question}: câu hỏi của người dùng
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    # Hàm ghép nội dung các document thành một chuỗi để đưa vào prompt
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Xây dựng RAG chain theo pipeline:
    # 1. {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #    - "context": lấy câu hỏi → retriever tìm docs → format_docs ghép thành string
    #    - "question": truyền câu hỏi gốc qua không thay đổi
    # 2. prompt: điền context và question vào template
    # 3. llm: gửi prompt đến LLM để sinh câu trả lời
    # 4. StrOutputParser(): chuyển response của LLM thành string thuần
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chạy thử với một số câu hỏi mẫu
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_rag_with_sources():
    """Demo RAG kèm thông tin nguồn — cho biết câu trả lời lấy từ tài liệu nào."""

    vectorstore = create_kb()

    # Lấy 3 chunk liên quan nhất thay vì 2
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt yêu cầu LLM nêu rõ nguồn tài liệu đã dùng
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):"""
    )

    # Hàm format đặc biệt: thêm số thứ tự và tên nguồn vào mỗi chunk
    # Ví dụ: "[1] langchain_knowledge_base.md:\n<nội dung chunk>"
    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    # Chain tương tự basic RAG nhưng dùng format_docs_with_sources
    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")


def demo_rag_with_fallback():
    """Demo RAG với fallback — xử lý khi câu hỏi nằm ngoài knowledge base."""

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Prompt có chỉ thị rõ ràng: nếu không có trong context thì phải nói không biết
    # → tránh LLM "bịa đặt" thông tin (hallucination)
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based ONLY on the following context.
If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Fallback:\n")

    questions = [
        "What is the pricing for LangSmith?",  # Có trong knowledge base → trả lời được
        "What is the stock price of OpenAI?",  # Không có → dùng fallback
        "How do I deploy LangChain to AWS?",   # Không có → dùng fallback
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_structured_rag():
    """Demo RAG với output có cấu trúc — trả về object Pydantic thay vì string."""

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Định nghĩa schema output bằng Pydantic
    # LLM sẽ buộc phải trả về đúng format này (structured output)
    class RAGResponse(BaseModel):
        """Structured RAG response."""

        answer: str = Field(description="The answer to the question")
        # Mức độ tự tin của LLM vào câu trả lời
        confidence: str = Field(description="high, medium, or low")
        # Danh sách các nguồn tài liệu đã tham chiếu
        sources_used: List[str] = Field(description="List of sources referenced")
        # Câu hỏi gợi ý tiếp theo liên quan đến chủ đề
        follow_up: str = Field(description="Suggested follow-up question")

    # with_structured_output: buộc LLM trả về đúng schema RAGResponse
    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template(
        """
Based on the context below, answer the question.

Context:
{context}

Question: {question}

Provide a structured response."""
    )

    # Format docs kèm tên nguồn trong ngoặc vuông
    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
            for doc in docs
        )

    # Chain này kết thúc bằng structured_llm thay vì llm | StrOutputParser()
    # → output là đối tượng RAGResponse, không phải string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    print("Structured RAG Demo:\n")
    result = rag_chain.invoke("What is LangGraph?")

    # Truy cập từng trường của structured output
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")


# ============================================================
# EXERCISE: Xây dựng hệ thống Q&A cho tài liệu tùy ý
# ============================================================
def exercise_document_qa():
    """
    BÀI TẬP: Xây dựng hệ thống Q&A hoàn chỉnh cho tài liệu, bao gồm:
    1. Nhận một đoạn văn bản làm input
    2. Tách và tạo embedding cho văn bản đó
    3. Cho phép đặt nhiều câu hỏi liên tiếp
    4. Trả về câu trả lời kèm điểm tự tin (confidence score)
    """

    class DocumentQA:
        def __init__(self, document: str, source_name: str = "document"):
            """
            Khởi tạo hệ thống Q&A với một tài liệu.
            
            Args:
                document: Nội dung văn bản cần tra cứu
                source_name: Tên định danh của tài liệu (dùng làm metadata)
            """
            # Bước 1: Tách tài liệu thành các chunk nhỏ
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            doc = Document(page_content=document, metadata={"source": source_name})
            chunks = splitter.split_documents([doc])

            # Bước 2: Tạo vector store từ các chunk
            # Lưu ý: không truyền persist_directory → dùng in-memory (không lưu ra đĩa)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            )
            # Retriever lấy 3 chunk liên quan nhất cho mỗi câu hỏi
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            # Khởi tạo LLM riêng cho instance này
            self.llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)

            # Prompt yêu cầu LLM tự đánh giá mức độ tự tin trước câu trả lời
            self.prompt = ChatPromptTemplate.from_template(
                """
Answer based on the context. Rate your confidence (high/medium/low).

Context: {context}
Question: {question}

Format: [Confidence: X] Answer"""
            )

            # Hàm ghép nội dung các doc thành string để đưa vào prompt
            def format_docs(docs):
                return "\n".join(d.page_content for d in docs)

            # Xây dựng chain hoàn chỉnh
            self.chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

        def ask(self, question: str) -> str:
            """Đặt câu hỏi và nhận câu trả lời từ tài liệu."""
            return self.chain.invoke(question)

    # Tài liệu test: các sự kiện về Python
    test_doc = """
    The Python programming language was created by Guido van Rossum.
    First released in 1991, Python emphasizes code readability.
    Python 3.12 was released in October 2023 with improved error messages.
    The language is named after Monty Python, not the snake.
    """

    # Tạo hệ thống Q&A với tài liệu test
    qa = DocumentQA(test_doc, "python_facts")

    print("Document Q&A System:\n")
    questions = [
        "Who created Python?",
        "When was Python 3.12 released?",
        "Why is Python named Python?",
    ]

    for q in questions:
        answer = qa.ask(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    # Bỏ comment dòng nào để chạy demo tương ứng:
    # demo_basic_rag()          # RAG cơ bản
    # demo_rag_with_sources()   # RAG kèm nguồn tài liệu
    # demo_rag_with_fallback()  # RAG với fallback khi không có thông tin
    # demo_structured_rag()     # RAG với output có cấu trúc (Pydantic)
    exercise_document_qa()      # Bài tập: hệ thống Q&A tùy chỉnh
