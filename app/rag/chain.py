"""
RAG Chain Module

Implements the RAG (Retrieval-Augmented Generation) chain that combines
document retrieval with LLM generation to answer questions.

Key Components:
- RAGChain: Orchestrates retrieval and generation workflow
- Source grounding enforcement to reduce hallucinations
- Chat history integration for conversational context
- Source attribution in responses
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from app.core.llm import LLMFactory


class RAGChain:
    """
    RAG Chain for question answering with retrieved context.
    
    Orchestrates the complete RAG workflow:
    1. Retrieve relevant documents using RAGRetriever
    2. Format documents into context string
    3. Create grounded prompt with context and question
    4. Generate answer using LLM
    5. Return answer with source documents
    
    Attributes:
        retriever: RAGRetriever instance for fetching documents
        llm: LangChain-compatible LLM for generating answers
        prompt_manager: Optional prompt template manager
        memory: Optional conversation memory for multi-turn chat
        return_source_documents: Whether to include sources in response
    
    Example:
        # Create chain
        chain = RAGChain(
            retriever=retriever,
            llm=LLMFactory.create()
        )
        
        # Ask a question
        result = chain.run("What is RAG?")
        print(result["answer"])
        
        # With chat history
        result = chain.run(
            "Tell me more",
            chat_history=[("What is RAG?", "RAG is...")]
        )
    """
    
    def __init__(
        self,
        retriever,
        llm,
        prompt_manager=None,
        memory=None,
        return_source_documents: bool = True,
        **kwargs
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: RAGRetriever instance
            llm: LangChain-compatible LLM
            prompt_manager: Optional prompt manager
            memory: Optional conversation memory
            return_source_documents: Include sources in response
            **kwargs: Additional configuration
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.memory = memory
        self.return_source_documents = return_source_documents
        
        # Store additional config
        self.config = kwargs
    
    def run(
        self,
        query: str,
        chat_history: Optional[List[tuple]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute RAG workflow to answer a question.
        
        This method:
        1. Validates input query
        2. Retrieves relevant documents
        3. Formats context from documents
        4. Creates grounded prompt
        5. Generates answer using LLM
        6. Returns answer with sources
        
        Args:
            query: User's question
            chat_history: Optional list of (question, answer) tuples
            **kwargs: Additional parameters (k, score_threshold, etc.)
            
        Returns:
            Dictionary containing:
            - answer: Generated answer string
            - source_documents: List of source Document objects
            - metadata: Additional info (query, confidence, etc.)
            
        Raises:
            ValueError: If query is empty or invalid
            
        Example:
            result = chain.run("What is machine learning?")
            print(result["answer"])
            for doc in result["source_documents"]:
                print(doc.metadata.get("source"))
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        # Step 1: Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=query,
            chat_history=chat_history,
            **kwargs
        )
        
        # Step 2: Handle case of no documents found
        if not documents:
            return {
                "answer": "I don't have enough information to answer this question. "
                         "No relevant documents were found in the knowledge base.",
                "source_documents": [],
                "metadata": {
                    "query": query,
                    "num_sources": 0,
                    "has_context": False
                }
            }
        
        # Step 3: Format documents into context string
        context = self._format_context(documents)
        
        # Step 4: Create grounded prompt
        prompt = self._create_prompt(
            query=query,
            context=context,
            chat_history=chat_history
        )
        
        # Step 5: Generate answer using LLM
        # LangChain LLMs have an invoke() method that returns AIMessage
        response = self.llm.invoke(prompt)
        
        # Extract answer text from response
        # AIMessage objects have a 'content' attribute
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Step 6: Build and return response
        result = {
            "answer": answer,
            "metadata": {
                "query": query,
                "num_sources": len(documents),
                "has_context": True
            }
        }
        
        # Include source documents if requested
        if self.return_source_documents:
            result["source_documents"] = documents
        
        return result
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.
        
        This method:
        - Extracts content from each document
        - Adds source attribution (optional)
        - Joins into single formatted string
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        # Format each document with optional source info
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            # Extract source filename if available
            source = doc.metadata.get("source", "Unknown")
            
            # Format: [Source N: filename]
            # Content...
            formatted = f"[Source {i}: {source}]\n{doc.page_content}"
            formatted_docs.append(formatted)
        
        # Join all documents with double newlines
        return "\n\n".join(formatted_docs)
    
    def _create_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[tuple]] = None
    ) -> str:
        """
        Create grounded prompt with context and query.
        
        This method enforces source grounding by instructing the LLM to:
        - Only use information from the provided context
        - Say "I don't know" if answer not in context
        - Not make up or hallucinate information
        
        Args:
            query: User's question
            context: Formatted context from documents
            chat_history: Optional conversation history
            
        Returns:
            Complete prompt string ready for LLM
        """
        # If prompt manager provided, use it
        if self.prompt_manager:
            pass
        
        # Build prompt with grounding instructions
        prompt_parts = []
        
        # System instructions (grounding)
        prompt_parts.append(
            "You are a helpful AI assistant.\n\n"
            "Grounding rules:\n"
            "- Use ONLY the information in the provided context.\n"
            "- If the answer is not in the context, reply with: I don't know.\n"
            "- Do not assume anything not explicitly stated in the context.\n"
            "- Do not make up facts or hallucinate.\n"
            "- If multiple sources apply, synthesize them."
        )
        
        # Add chat history if provided
        if chat_history:
            history_text = "\n".join([
                f"Human: {q}\nAssistant: {a}"
                for q, a in chat_history
            ])
            prompt_parts.append(f"\nPrevious conversation:\n{history_text}")
        
        # Add context
        prompt_parts.append(f"\n---\nContext:\n{context}\n---")
        
        # Add user question
        prompt_parts.append(f"\nQuestion: {query}")
        
        # Add explicit reminder (expected by tests)
        prompt_parts.append(
            "\nRemember: Use only the context above. If you cannot answer from it, reply with: I don't know."
        )
        
        return "\n".join(prompt_parts)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_rag_chain(
    retriever,
    llm=None,
    prompt_manager=None,
    return_source_documents: bool = True,
    **kwargs
) -> RAGChain:
    """
    Convenience function to create a RAG chain.
    
    This function simplifies chain creation by:
    - Auto-creating LLM if not provided
    - Setting sensible defaults
    - Providing clean interface
    
    Args:
        retriever: RAGRetriever instance
        llm: Optional LLM instance (auto-created if None)
        prompt_manager: Optional prompt manager
        return_source_documents: Include sources in response
        **kwargs: Additional configuration
        
    Returns:
        Configured RAGChain instance
        
    Example:
        # Minimal usage (auto-create LLM)
        chain = create_rag_chain(retriever)
        
        # With custom LLM
        from app.core.llm import LLMFactory
        chain = create_rag_chain(
            retriever=retriever,
            llm=LLMFactory.create()
        )
    """
    # Auto-create LLM if not provided
    if llm is None:
        llm = LLMFactory.create()
    
    return RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_manager=prompt_manager,
        return_source_documents=return_source_documents,
        **kwargs
    )
