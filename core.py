import streamlit as st

st.set_page_config(
    page_title="Neura-IQ Multimodal AI Research Assistant",
    page_icon="🚀",
    layout="wide"
)

import os
import base64
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import shutil

import chromadb
from chromadb.config import Settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from PIL import Image
import numpy as np

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    T5ForConditionalGeneration, T5Tokenizer,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import random
from typing import Generator
import gc

# Authentication imports
from auth_utils import is_authenticated, get_current_user, is_admin
from auth_ui import render_login_page, render_user_management, render_sidebar_user_info, render_back_to_main_button

PDFS_DIRECTORY = 'data/pdfs/'
FIGURES_DIRECTORY = 'data/figures/'
TABLES_DIRECTORY = 'data/tables/'
SUMMARIES_DIRECTORY = 'data/summaries/'
CHROMADB_PATH = 'data/chromadb/'
METADATA_PATH = 'data/metadata.json'
FINETUNED_MODEL_PATH = 'data/finetuned_t5_lora/'
QA_DATASET_PATH = 'data/qa_dataset.json'
MODEL_CONFIG_PATH = 'data/model_config.json'

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_WORKERS = min(8, multiprocessing.cpu_count()) 
BATCH_SIZE = 10

for directory in [PDFS_DIRECTORY, FIGURES_DIRECTORY, TABLES_DIRECTORY, 
                 SUMMARIES_DIRECTORY, CHROMADB_PATH]:
    Path(directory).mkdir(parents=True, exist_ok=True)

@dataclass
class DocumentChunk:
    """Enhanced document chunk with multimodal support"""
    content: str
    chunk_type: str  
    summary: str
    source_file: str
    chunk_id: int
    raw_data_path: Optional[str] = None
    position_metadata: Optional[Dict] = None
    table_data: Optional[pd.DataFrame] = None
    image_path: Optional[str] = None

# Templates 
MULTIMODAL_RAG_TEMPLATE = """
You are an advanced multimodal AI assistant. Use the retrieved context to answer the question comprehensively.

Context includes:
- Text chunks with summaries
- Table data with summaries  
- Image descriptions with visual content

Question: {question}

Retrieved Context:
{context}

Instructions:
- Synthesize information from ALL modalities (text, tables, images)
- Reference specific sources when making claims
- If images or tables are relevant, describe what they show
- Keep answer comprehensive but concise
- If you cannot answer based on the provided context, say so clearly

Answer:
"""

TEXT_SUMMARIZATION_TEMPLATE = """
Summarize the following text chunk in 2-3 concise sentences. Focus on key information, facts, and concepts that would be useful for question-answering:

Text: {text}

Summary:
"""

TABLE_SUMMARIZATION_TEMPLATE = """
Analyze and summarize this table data. Describe what the table shows, key patterns, and important values:

Table Data:
{table_data}

Table Summary:
"""

IMAGE_ANALYSIS_TEMPLATE = """
Analyze this image comprehensively. Describe:
1. Any text content (OCR)
2. Charts, graphs, or data visualizations
3. Key visual elements and their relationships
4. Important information for question-answering

Provide a detailed but concise description:
"""

GENERAL_CHAT_TEMPLATE = """
You are a helpful and knowledgeable AI assistant. Respond to the user's question or request in a clear, informative, and engaging way.

User: {question}
"""

class EnhancedMultimodalRAG:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.chroma_client = None
        self.collection = None
        self.initialize_models()
        self.initialize_chromadb()
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self.is_finetuned_loaded = False
        self.base_t5_model = None
        self.base_t5_tokenizer = None
    
    @st.cache_resource
    def initialize_models(_self):
        """Initialize models with error handling"""
        try:
            import ollama
            ollama.list()
            _self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            _self.llm = OllamaLLM(model="llava:13b", temperature=0.1, num_ctx=4096)
            test_response = _self.llm.invoke("Test")
            if not test_response:
                raise ValueError("LLM returned empty response")
        except Exception as e:
            st.error(f"Model initialization failed: {str(e)}")
            st.error("""
            1. Ensure Ollama is running (`ollama serve`)
            2. Verify models are downloaded:
               - `ollama pull llava:13b`
               - `ollama pull nomic-embed-text`
            """)
            st.stop()
    
    def initialize_chromadb(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMADB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection("multimodal_rag")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="multimodal_rag",
                    metadata={"description": "Enhanced multimodal RAG collection"}
                )
                
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            st.stop()
    
    def extract_pdf_elements(self, file_path: str, filename: str) -> Tuple[List[str], List[Dict], List[str]]:
        """Enhanced PDF extraction with better element separation and bug fixes"""
        with st.spinner(f"🔍 Extracting elements from {filename}..."):
            pdf_name = filename.replace('.pdf', '')
            figures_dir = os.path.join(FIGURES_DIRECTORY, pdf_name)
            tables_dir = os.path.join(TABLES_DIRECTORY, pdf_name)
            
            Path(figures_dir).mkdir(parents=True, exist_ok=True)
            Path(tables_dir).mkdir(parents=True, exist_ok=True)
    
            try:
                elements = partition_pdf(
                    file_path,
                    strategy=PartitionStrategy.HI_RES,
                    extract_image_block_types=["Image", "Table"],
                    extract_image_block_output_dir=figures_dir,
                    extract_image_block_to_payload=False,
                    infer_table_structure=True
                )
                
                text_elements = []
                table_elements = []
                image_paths = []
                
                for i, element in enumerate(elements):
                    try:
                        element_dict = element.to_dict()
                        
                        if hasattr(element, 'category'):
                            if element.category == "Table":
                                table_data = {
                                    'content': element.text if hasattr(element, 'text') else '',
                                    'metadata': element_dict.get('metadata', {}),
                                    'position': i,
                                    'source': filename
                                }
                                
                                if hasattr(element, 'metadata') and element.metadata:
                                    metadata_dict = element.metadata
                                    if hasattr(metadata_dict, '__dict__'):
                                        metadata_dict = metadata_dict.__dict__
                                    elif not isinstance(metadata_dict, dict):
                                        metadata_dict = {}
                                    
                                    if 'text_as_html' in metadata_dict:
                                        try:
                                            df = pd.read_html(metadata_dict['text_as_html'])[0]
                                            table_path = os.path.join(tables_dir, f"table_{i}.csv")
                                            df.to_csv(table_path, index=False)
                                            table_data['csv_path'] = table_path
                                            table_data['dataframe'] = df
                                        except Exception as e:
                                            st.warning(f"Could not parse table HTML: {str(e)}")
                                
                                table_elements.append(table_data)
                                
                            elif element.category not in ["Image", "Table"]:
                                if hasattr(element, 'text') and element.text and element.text.strip():
                                    text_elements.append(element.text.strip())
                    
                    except Exception as e:
                        st.warning(f"Error processing element {i}: {str(e)}")
                        continue
                
                if os.path.exists(figures_dir):
                    try:
                        image_files = [f for f in os.listdir(figures_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        image_paths = [os.path.join(figures_dir, f) for f in image_files]
                    except Exception as e:
                        st.warning(f"Error reading images directory: {str(e)}")
                        image_paths = []
                
                return text_elements, table_elements, image_paths
                
            except Exception as e:
                st.error(f"Error extracting from {filename}: {str(e)}")
                return [], [], []
    
    def summarize_text_chunk(self, text: str) -> str:
        """Generate summary for text chunk"""
        try:
            prompt = ChatPromptTemplate.from_template(TEXT_SUMMARIZATION_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"text": text[:2000]})  # Limiting input size
            return response.strip()
        except Exception as e:
            st.warning(f"Error summarizing text: {str(e)}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def summarize_table(self, table_data: Dict) -> str:
        """Generate summary for table"""
        try:
            if 'dataframe' in table_data and table_data['dataframe'] is not None:
                df = table_data['dataframe']
                table_str = df.to_string(max_rows=10, max_cols=10)
            else:
                table_str = table_data.get('content', '')[:1000]
            
            prompt = ChatPromptTemplate.from_template(TABLE_SUMMARIZATION_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"table_data": table_str})
            return response.strip()
        except Exception as e:
            st.warning(f"Error summarizing table: {str(e)}")
            return f"Table from {table_data.get('source', 'unknown source')}"
    
    def analyze_image(self, image_path: str) -> str:
        """Generate description for image using multimodal LLM"""
        try:
            response = self.llm.invoke(
                IMAGE_ANALYSIS_TEMPLATE,
                images=[image_path]
            )
            return response.strip()
        except Exception as e:
            st.warning(f"Error analyzing image {image_path}: {str(e)}")
            return f"Image: {os.path.basename(image_path)}"
    
    def process_elements_parallel(self, text_elements: List[str], table_elements: List[Dict], 
                                image_paths: List[str], filename: str) -> List[DocumentChunk]:
        """Process all elements in parallel for better performance"""
        chunks = []
        
        with st.spinner(f"🧠 Processing {len(text_elements + table_elements + image_paths)} elements..."):
            
            if text_elements:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                all_text_chunks = []
                for text in text_elements:
                    if len(text) > CHUNK_SIZE:
                        text_chunks = text_splitter.split_text(text)
                        all_text_chunks.extend(text_chunks)
                    else:
                        all_text_chunks.append(text)
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    text_futures = {
                        executor.submit(self.summarize_text_chunk, chunk): (i, chunk)
                        for i, chunk in enumerate(all_text_chunks)
                    }
                    
                    progress_bar = st.progress(0)
                    completed = 0
                    
                    for future in as_completed(text_futures):
                        i, chunk = text_futures[future]
                        try:
                            summary = future.result()
                            chunks.append(DocumentChunk(
                                content=chunk,
                                chunk_type='text',
                                summary=summary,
                                source_file=filename,
                                chunk_id=len(chunks),
                                position_metadata={'text_index': i}
                            ))
                        except Exception as e:
                            st.warning(f"Failed to process text chunk {i}: {str(e)}")
                        
                        completed += 1
                        progress_bar.progress(completed / len(text_futures))
            
            if table_elements:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    table_futures = {
                        executor.submit(self.summarize_table, table): (i, table)
                        for i, table in enumerate(table_elements)
                    }
                    
                    for future in as_completed(table_futures):
                        i, table = table_futures[future]
                        try:
                            summary = future.result()
                            chunks.append(DocumentChunk(
                                content=table.get('content', ''),
                                chunk_type='table',
                                summary=summary,
                                source_file=filename,
                                chunk_id=len(chunks),
                                raw_data_path=table.get('csv_path'),
                                table_data=table.get('dataframe'),
                                position_metadata={'table_index': i}
                            ))
                        except Exception as e:
                            st.warning(f"Failed to process table {i}: {str(e)}")
            
            if image_paths:
                with ThreadPoolExecutor(max_workers=min(4, MAX_WORKERS)) as executor:  # Limit for memory
                    image_futures = {
                        executor.submit(self.analyze_image, path): (i, path)
                        for i, path in enumerate(image_paths)
                    }
                    
                    for future in as_completed(image_futures):
                        i, path = image_futures[future]
                        try:
                            description = future.result()
                            chunks.append(DocumentChunk(
                                content=description,
                                chunk_type='image',
                                summary=description,  
                                source_file=filename,
                                chunk_id=len(chunks),
                                image_path=path,
                                position_metadata={'image_index': i}
                            ))
                        except Exception as e:
                            st.warning(f"Failed to process image {i}: {str(e)}")
        
        return chunks
    
    def embed_and_store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Embed chunks and store in ChromaDB"""
        if not chunks:
            return False
        
        try:
            with st.spinner(f"🔢 Generating embeddings for {len(chunks)} chunks..."):
                documents = []
                metadatas = []
                ids = []
                
                for chunk in chunks:
                    documents.append(chunk.summary)
                    
                    metadata = {
                        'source_file': chunk.source_file,
                        'chunk_type': chunk.chunk_type,
                        'chunk_id': chunk.chunk_id,
                        'content': chunk.content[:1000],  # Truncating for storage
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    if chunk.raw_data_path:
                        metadata['raw_data_path'] = chunk.raw_data_path
                    if chunk.image_path:
                        metadata['image_path'] = chunk.image_path
                    if chunk.position_metadata:
                        metadata.update(chunk.position_metadata)
                    
                    metadatas.append(metadata)
                    ids.append(f"{chunk.source_file}_{chunk.chunk_type}_{chunk.chunk_id}")
                
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                return True
                
        except Exception as e:
            st.error(f"Error storing chunks: {str(e)}")
            return False
    
    def process_pdf(self, file_path: str, filename: str) -> bool:
        """Complete PDF processing pipeline"""
        try:
            text_elements, table_elements, image_paths = self.extract_pdf_elements(file_path, filename)
            
            if not any([text_elements, table_elements, image_paths]):
                st.warning(f"No content extracted from {filename}")
                return False
            
            st.success(f"📊 Extracted: {len(text_elements)} text blocks, {len(table_elements)} tables, {len(image_paths)} images")
            
            chunks = self.process_elements_parallel(text_elements, table_elements, image_paths, filename)
            
            if not chunks:
                st.warning(f"No chunks created from {filename}")
                return False
            
            success = self.embed_and_store_chunks(chunks)
            
            if success:
                st.success(f"✅ Successfully processed {filename}: {len(chunks)} chunks stored")
                return True
            else:
                st.error(f"Failed to store chunks for {filename}")
                return False
                
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            return False
    
    def multi_vector_retrieve(self, query: str, k: int = 6) -> List[Dict]:
        """Enhanced retrieval with multi-vector approach"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                return []
            
            retrieved_chunks = []
            
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                
                chunk_data = {
                    'summary': results['documents'][0][i],
                    'content': metadata.get('content', ''),
                    'chunk_type': metadata.get('chunk_type', 'text'),
                    'source_file': metadata.get('source_file', ''),
                    'distance': results['distances'][0][i]
                }
                
                if metadata.get('chunk_type') == 'table' and metadata.get('raw_data_path'):
                    try:
                        df = pd.read_csv(metadata['raw_data_path'])
                        chunk_data['table_data'] = df
                    except:
                        pass
                
                if metadata.get('chunk_type') == 'image' and metadata.get('image_path'):
                    chunk_data['image_path'] = metadata['image_path']
                
                retrieved_chunks.append(chunk_data)
            
            return retrieved_chunks
            
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_multimodal_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate answer using multimodal context"""
        if not retrieved_chunks:
            return "I don't have relevant information to answer your question."
        
        try:
            context_parts = []
            images_for_llm = []
            
            for i, chunk in enumerate(retrieved_chunks):
                source = chunk.get('source_file', 'Unknown')
                chunk_type = chunk.get('chunk_type', 'text')
                
                context_parts.append(f"\n--- Source {i+1}: {source} ({chunk_type.upper()}) ---")
                context_parts.append(f"Summary: {chunk['summary']}")
                
                if chunk_type == 'text':
                    context_parts.append(f"Full Text: {chunk['content']}")
                
                elif chunk_type == 'table' and 'table_data' in chunk:
                    df = chunk['table_data']
                    context_parts.append(f"Table Data:\n{df.to_string(max_rows=20)}")
                
                elif chunk_type == 'image' and 'image_path' in chunk:
                    context_parts.append(f"Image Description: {chunk['summary']}")
                    if os.path.exists(chunk['image_path']):
                        images_for_llm.append(chunk['image_path'])
            
            context = "\n".join(context_parts)
            
            prompt = ChatPromptTemplate.from_template(MULTIMODAL_RAG_TEMPLATE)
            
            with st.spinner("🤖 Generating multimodal answer..."):
                if images_for_llm:
                    response = self.llm.invoke(
                        prompt.format(question=query, context=context),
                        images=images_for_llm[:3]  # Limiting to 3 images for performance
                    )
                else:
                    chain = prompt | self.llm
                    response = chain.invoke({"question": query, "context": context})
                
                return response
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            if count > 0:
                sample_results = self.collection.query(
                    query_texts=["sample"],
                    n_results=min(100, count),
                    include=['metadatas']
                )
                
                type_counts = {}
                sources = set()
                
                for metadata in sample_results['metadatas'][0]:
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                    sources.add(metadata.get('source_file', 'unknown'))
                
                return {
                    'total_chunks': count,
                    'type_distribution': type_counts,
                    'unique_documents': len(sources),
                    'document_names': list(sources)
                }
            
            return {'total_chunks': 0, 'type_distribution': {}, 'unique_documents': 0, 'document_names': []}
            
        except Exception as e:
            st.error(f"Error getting collection stats: {str(e)}")
            return {'total_chunks': 0, 'type_distribution': {}, 'unique_documents': 0, 'document_names': []}
        
    def clear_all_data(self):
        """Clear all data and reset system"""
        try:
            self.chroma_client.delete_collection("multimodal_rag")
            self.collection = self.chroma_client.create_collection(
                name="multimodal_rag",
                metadata={"description": "Enhanced multimodal RAG collection"}
            )
            
            for directory in [PDFS_DIRECTORY, FIGURES_DIRECTORY, TABLES_DIRECTORY, SUMMARIES_DIRECTORY]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                    Path(directory).mkdir(parents=True, exist_ok=True)
            
            if os.path.exists(METADATA_PATH):
                os.remove(METADATA_PATH)
            
            return True
            
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
            return False
        
    def general_multimodal_chat(self, query: str, uploaded_image=None) -> str:
        """Handle general multimodal chat without RAG"""
        try:
            prompt = ChatPromptTemplate.from_template(GENERAL_CHAT_TEMPLATE)
            
            with st.spinner("🤖 Generating response..."):
                if uploaded_image is not None:
                    temp_image_path = "temp_uploaded_image.png"
                    uploaded_image.save(temp_image_path)
                    
                    response = self.llm.invoke(
                        prompt.format(question=query),
                        images=[temp_image_path]
                    )
                    
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                else:
                    chain = prompt | self.llm
                    response = chain.invoke({"question": query})
                
                return response
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating the response."
    
    def load_base_t5_model(_self):
        """Load base T5-small model for fine-tuning and inference"""
        try:
            model_name = "google/flan-t5-small"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # for CPU optimization
                device_map="cpu"
            )
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading T5 model: {str(e)}")
            return None, None
    
    def generate_qa_pairs_from_chunks(self, max_pairs: int = 100, 
                                    pairs_per_chunk: int = 2) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from stored document chunks using LLaVA
        
        Args:
            max_pairs: Maximum number of Q&A pairs to generate
            pairs_per_chunk: Number of Q&A pairs per chunk
            
        Returns:
            List of {"question": str, "answer": str} dictionaries
        """
        try:
            all_results = self.collection.query(
                query_texts=["general knowledge"],
                n_results=min(max_pairs // pairs_per_chunk, 50),
                include=['documents', 'metadatas']
            )
            
            if not all_results['documents'][0]:
                st.warning("No documents found in the index")
                return []
            
            qa_pairs = []
            

            qa_template = """
            Based on the following content, generate {num_pairs} diverse question-answer pairs. 
            Make questions specific and answerable from the content. Format as JSON:
            
            Content: {content}
            
            Generate exactly {num_pairs} Q&A pairs in this JSON format:
            [
                {{"question": "What is...", "answer": "The answer is..."}},
                {{"question": "How does...", "answer": "It works by..."}}
            ]
            
            JSON Response:
            """
            
            with st.spinner(f"🧠 Generating Q&A pairs from {len(all_results['documents'][0])} chunks..."):
                progress_bar = st.progress(0)
                
                for i, (doc, metadata) in enumerate(zip(all_results['documents'][0], 
                                                       all_results['metadatas'][0])):
                    try:

                        content = doc[:1000]  # Limiting content length
                        prompt = qa_template.format(
                            content=content, 
                            num_pairs=pairs_per_chunk
                        )
                        
                        response = self.llm.invoke(prompt)
                        
                        try:
                            json_start = response.find('[')
                            json_end = response.rfind(']') + 1
                            
                            if json_start != -1 and json_end != 0:
                                json_str = response[json_start:json_end]
                                pairs = json.loads(json_str)
                                
                                for pair in pairs:
                                    if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                                        enhanced_pair = {
                                            'question': pair['question'].strip(),
                                            'answer': pair['answer'].strip(),
                                            'source_file': metadata.get('source_file', 'unknown'),
                                            'chunk_type': metadata.get('chunk_type', 'text')
                                        }
                                        qa_pairs.append(enhanced_pair)
                                        
                                        if len(qa_pairs) >= max_pairs:
                                            break
                            
                        except json.JSONDecodeError:
                            st.warning(f"Could not parse JSON from chunk {i}")
                            continue
                    
                    except Exception as e:
                        st.warning(f"Error processing chunk {i}: {str(e)}")
                        continue
                    
                    progress_bar.progress((i + 1) / len(all_results['documents'][0]))
                    
                    if len(qa_pairs) >= max_pairs:
                        break
                
                progress_bar.empty()
            
            unique_qa_pairs = []
            seen_questions = set()
            
            for pair in qa_pairs:
                q_lower = pair['question'].lower().strip()
                if q_lower not in seen_questions and len(pair['question']) > 10:
                    seen_questions.add(q_lower)
                    unique_qa_pairs.append(pair)
            
            st.success(f"✅ Generated {len(unique_qa_pairs)} unique Q&A pairs")
            return unique_qa_pairs
            
        except Exception as e:
            st.error(f"Error generating Q&A pairs: {str(e)}")
            return []
    
    def fine_tune_flan_t5(self, qa_pairs: List[Dict[str, str]], 
                         num_epochs: int = 3, 
                         learning_rate: float = 3e-4) -> bool:
        """
        Fine-tune FLAN-T5 small model using LoRA on generated Q&A pairs
        
        Args:
            qa_pairs: List of Q&A dictionaries
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(qa_pairs) < 10:
                st.error("Need at least 10 Q&A pairs for fine-tuning")
                return False
            
            base_model, tokenizer = self.load_base_t5_model()
            if base_model is None:
                return False
            
            with st.spinner("🔧 Setting up LoRA configuration..."):
                # Configuring LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=16,  # Low rank
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
                )
                
                model = get_peft_model(base_model, lora_config)
                model.print_trainable_parameters()
            
            with st.spinner("📊 Preparing training dataset..."):
                train_data = []
                for pair in qa_pairs:
                    input_text = f"question: {pair['question']}"
                    target_text = pair['answer']
                    train_data.append({
                        'input_text': input_text,
                        'target_text': target_text
                    })

                dataset = Dataset.from_list(train_data)
                
                def tokenize_function(examples):
                    model_inputs = tokenizer(
                        examples['input_text'],
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    targets = tokenizer(
                        examples['target_text'],
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    model_inputs['labels'] = targets['input_ids']
                    return model_inputs
                
                tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments optimized for CPU
            training_args = TrainingArguments(
                output_dir=FINETUNED_MODEL_PATH,
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=2,  # Small batch for CPU
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                dataloader_num_workers=min(4, multiprocessing.cpu_count()//4),
                fp16=False,  # CPU doesn't support fp16
                dataloader_pin_memory=False,
                report_to=None,  # Disable wandb/tensorboard
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
            )
            
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                return_tensors="pt"
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            with st.spinner(f"🚀 Fine-tuning model ({num_epochs} epochs)..."):
                progress_container = st.container()
                
                class ProgressCallback:
                    def __init__(self, container):
                        self.container = container
                        self.progress_bar = None
                        self.status_text = None
                    
                    def on_epoch_begin(self, args, state, control, **kwargs):
                        if self.progress_bar is None:
                            self.progress_bar = self.container.progress(0)
                            self.status_text = self.container.empty()
                        
                        epoch = state.epoch
                        self.status_text.text(f"Epoch {epoch+1}/{num_epochs}")
                        self.progress_bar.progress((epoch) / num_epochs)
                    
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if logs and 'loss' in logs:
                            if self.status_text:
                                self.status_text.text(f"Epoch {state.epoch:.0f}/{num_epochs} - Loss: {logs['loss']:.4f}")
                
                progress_callback = ProgressCallback(progress_container)
                trainer.add_callback(progress_callback)
                
                trainer.train()
            
            with st.spinner("💾 Saving fine-tuned model..."):
                Path(FINETUNED_MODEL_PATH).mkdir(parents=True, exist_ok=True)
                
                model.save_pretrained(FINETUNED_MODEL_PATH)
                tokenizer.save_pretrained(FINETUNED_MODEL_PATH)
                
                config = {
                    'base_model': 'google/flan-t5-small',
                    'num_qa_pairs': len(qa_pairs),
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'fine_tuned_at': datetime.now().isoformat(),
                    'lora_config': lora_config.to_dict()
                }
                
                with open(MODEL_CONFIG_PATH, 'w') as f:
                    json.dump(config, f, indent=2)
            
            del model, trainer, tokenized_dataset
            gc.collect()
            
            st.success("✅ Fine-tuning completed successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error during fine-tuning: {str(e)}")
            return False
    
    def load_finetuned_model(self) -> bool:
        """Load the fine-tuned LoRA model for inference"""
        try:
            if not os.path.exists(FINETUNED_MODEL_PATH):
                st.warning("No fine-tuned model found")
                return False
            
            with st.spinner("📥 Loading fine-tuned model..."):
                base_model, tokenizer = self.load_base_t5_model()
                if base_model is None:
                    return False
                
                model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
                
                self.finetuned_model = model
                self.finetuned_tokenizer = tokenizer
                self.base_t5_model = base_model
                self.base_t5_tokenizer = tokenizer
                self.is_finetuned_loaded = True
                
                st.success("✅ Fine-tuned model loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"Error loading fine-tuned model: {str(e)}")
            return False
    
    def generate_flan_response(self, question: str, use_finetuned: bool = True) -> str:
        """
        Generate response using T5 model (base or fine-tuned)
        
        Args:
            question: Input question
            use_finetuned: Whether to use fine-tuned model
            
        Returns:
            Generated response
        """
        try:
            if use_finetuned and self.is_finetuned_loaded:
                model = self.finetuned_model
                tokenizer = self.finetuned_tokenizer
                model_name = "Fine-tuned T5"
            else:
                if self.base_t5_model is None:
                    self.base_t5_model, self.base_t5_tokenizer = self.load_base_t5_model()
                model = self.base_t5_model
                tokenizer = self.base_t5_tokenizer
                model_name = "Base T5"
            
            if model is None:
                return "Error: Model not available"
            
            input_text = f"question: {question}"
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=256,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return f"**{model_name}:** {response}"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
def get_file_hash(file_content):
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def load_metadata():
    """Load processing metadata"""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_metadata(metadata):
    """Save processing metadata"""
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

def upload_pdf(file):
    """Save uploaded PDF"""
    try:
        file_path = os.path.join(PDFS_DIRECTORY, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error uploading PDF: {str(e)}")
        return None

def render_mode_selector():
    """Render the main mode selection interface"""
    st.title("🚀 Neura-IQ Multimodal AI Research Assistant")
    st.markdown("**Choose your interaction mode:**")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 Create New Index", use_container_width=True, type="primary"):
            st.session_state.current_mode = "create_index"
            st.rerun()
        st.markdown("Upload and process your documents to build a searchable knowledge base")
        
        if st.button("🧠 Fine-Tune Model", use_container_width=True, type="secondary"):
            st.session_state.current_mode = "finetune"
            st.rerun()
        st.markdown("Generate Q&A pairs from your index and fine-tune a lightweight T5 model")
    
    with col2:
        if st.button("💬 Chat with Index", use_container_width=True, type="secondary"):
            st.session_state.current_mode = "chat_index"
            st.rerun()
        st.markdown("Ask questions about your processed documents using multimodal RAG")
        
        if st.button("🤖 General LLM Chat", use_container_width=True, type="secondary"):
            st.session_state.current_mode = "general_chat"
            st.rerun()
        st.markdown("General multimodal chat with image analysis capabilities")

def render_create_index_mode(rag_system):
    """Render the document processing interface"""
    st.header("📚 Create New Index")
    st.markdown("Upload and process PDF documents to build your knowledge base")

    if st.button("← Back to Main Menu"):
        st.session_state.current_mode = "main"
        st.rerun()
    
    stats = rag_system.get_collection_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", stats['total_chunks'])
    with col2:
        st.metric("Documents", stats['unique_documents'])
    with col3:
        text_count = stats['type_distribution'].get('text', 0)
        st.metric("Text Chunks", text_count)
    with col4:
        image_count = stats['type_distribution'].get('image', 0)
        table_count = stats['type_distribution'].get('table', 0)
        st.metric("Images + Tables", image_count + table_count)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type="pdf",
            help="Upload PDF with text, tables, and images"
        )
        
        if uploaded_file:
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            metadata = load_metadata()
            is_duplicate = any(
                info.get('file_hash') == file_hash 
                for info in metadata.values()
            )
            
            if is_duplicate:
                st.warning("⚠️ This file appears to be already processed")
            else:
                if st.button("🔄 Process Document", type="primary"):
                    file_path = upload_pdf(uploaded_file)
                    if file_path:
                        success = rag_system.process_pdf(file_path, uploaded_file.name)
                        
                        if success:
                            metadata = load_metadata()
                            metadata[uploaded_file.name] = {
                                "file_hash": file_hash,
                                "processed_at": datetime.now().isoformat(),
                                "file_size": len(file_content)
                            }
                            save_metadata(metadata)
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                            st.rerun()
    
    with col2:
        st.markdown("**🔧 Index Management**")
        
        if stats['total_chunks'] > 0:
            if st.button("🗑️ Clear All Data", help="Reset entire system"):
                if st.session_state.get('confirm_clear'):
                    if rag_system.clear_all_data():
                        st.session_state.confirm_clear = False
                        st.success("✅ System cleared!")
                        st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.error("⚠️ Click again to confirm!")
        
        if st.button("📊 View Analytics"):
            st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
    
    if stats['unique_documents'] > 0:
        st.markdown("---")
        st.markdown("### 📋 Processed Documents")
        
        for doc_name in stats['document_names']:
            with st.expander(f"📄 {doc_name}"):
                metadata = load_metadata().get(doc_name, {})
                if metadata:
                    st.write(f"**Processed:** {metadata.get('processed_at', 'Unknown')}")
                    st.write(f"**File Size:** {metadata.get('file_size', 0):,} bytes")
    
    if st.session_state.get('show_analytics', False) and stats['total_chunks'] > 0:
        st.markdown("---")
        st.markdown("### 📊 Index Analytics")
        
        type_counts = stats['type_distribution']
        if type_counts:
            st.markdown("**Content Distribution:**")
            for content_type, count in type_counts.items():
                percentage = (count / stats['total_chunks']) * 100
                st.write(f"• {content_type.title()}: {count} ({percentage:.1f}%)")

def render_finetune_mode(rag_system):
    """Render the fine-tuning interface"""
    st.header("🧠 Fine-Tune Model from Index")
    st.markdown("Generate Q&A pairs from your documents and fine-tune a lightweight T5 model")
    
    if st.button("← Back to Main Menu"):
        st.session_state.current_mode = "main"
        st.rerun()
    
    stats = rag_system.get_collection_stats()
    if stats['total_chunks'] == 0:
        st.warning("⚠️ No documents in index. Go to 'Create New Index' first!")
        return
    
    st.success(f"📊 Ready to process {stats['total_chunks']} chunks from {stats['unique_documents']} documents")
    
    tab1, tab2, tab3 = st.tabs(["📝 Generate Q&A", "🚀 Fine-Tune", "🧪 Test Model"])
    
    with tab1:
        st.markdown("### Step 1: Generate Q&A Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            max_pairs = st.slider("Max Q&A pairs", 20, 200, 100)
            pairs_per_chunk = st.slider("Pairs per chunk", 1, 5, 2)
        
        with col2:
            st.info(f"Will process ~{min(max_pairs // pairs_per_chunk, 50)} chunks")
        
        if st.button("🧠 Generate Q&A Pairs", type="primary"):
            qa_pairs = rag_system.generate_qa_pairs_from_chunks(max_pairs, pairs_per_chunk)
            
            if qa_pairs:
                st.session_state.qa_pairs = qa_pairs
                
                st.markdown("### 📋 Generated Q&A Samples")
                for i, pair in enumerate(qa_pairs[:3]):
                    with st.expander(f"Sample {i+1} - From: {pair['source_file']}"):
                        st.markdown(f"**Q:** {pair['question']}")
                        st.markdown(f"**A:** {pair['answer']}")
                        st.caption(f"Type: {pair['chunk_type']}")
                
                st.markdown("### 💾 Download Dataset")
                qa_json = json.dumps(qa_pairs, indent=2)
                st.download_button(
                    label="📥 Download Q&A Dataset (JSON)",
                    data=qa_json,
                    file_name=f"qa_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        st.markdown("---")
        st.markdown("### 📤 Or Upload Existing Dataset")
        uploaded_qa = st.file_uploader("Upload Q&A JSON file", type="json")
        if uploaded_qa:
            try:
                qa_data = json.load(uploaded_qa)
                st.session_state.qa_pairs = qa_data
                st.success(f"✅ Loaded {len(qa_data)} Q&A pairs")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.markdown("### Step 2: Fine-Tune T5 Model")
        
        if 'qa_pairs' not in st.session_state:
            st.warning("⚠️ Generate or upload Q&A pairs first!")
            return
        
        qa_pairs = st.session_state.qa_pairs
        st.success(f"📊 Ready to fine-tune with {len(qa_pairs)} Q&A pairs")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.slider("Training epochs", 1, 10, 3)
            learning_rate = st.select_slider(
                "Learning rate", 
                options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                value=3e-4,
                format_func=lambda x: f"{x:.0e}"
            )
        
        with col2:
            st.info("**Estimated time:**")
            estimated_time = (len(qa_pairs) * num_epochs) / 100  # Rough estimate
            st.write(f"~{estimated_time:.1f} minutes")
            st.write("(CPU-based training)")
        
        if st.button("🚀 Start Fine-Tuning", type="primary"):
            success = rag_system.fine_tune_flan_t5(qa_pairs, num_epochs, learning_rate)
            if success:
                st.session_state.model_finetuned = True
                st.balloons()
    
    with tab3:
        st.markdown("### Step 3: Test Your Models")
        
        col1, col2 = st.columns(2)
        with col1:
            base_available = st.checkbox("Use Base T5", value=True)
        with col2:
            finetuned_available = os.path.exists(FINETUNED_MODEL_PATH)
            if finetuned_available:
                use_finetuned = st.checkbox("Use Fine-tuned T5", value=True)
                if use_finetuned and not rag_system.is_finetuned_loaded:
                    if st.button("📥 Load Fine-tuned Model"):
                        rag_system.load_finetuned_model()
            else:
                st.info("Fine-tuned model not available")
                use_finetuned = False
        
        # Model info
        if finetuned_available:
            try:
                with open(MODEL_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                st.success(f"Fine-tuned model available (trained on {config['num_qa_pairs']} pairs)")
            except:
                st.warning("Fine-tuned model found but no config available")
        
        st.markdown("### 🧪 Compare Model Responses")
        
        if 'qa_pairs' in st.session_state:
            sample_questions = [pair['question'] for pair in st.session_state.qa_pairs[:5]]
            st.markdown("**Sample questions from your dataset:**")
            for i, q in enumerate(sample_questions):
                if st.button(f"💬 {q[:50]}...", key=f"sample_q_{i}"):
                    st.session_state.test_question = q
        
        test_question = st.text_input(
            "Ask a question:",
            value=st.session_state.get('test_question', ''),
            placeholder="Enter your question here..."
        )
        
        if test_question and st.button("🔍 Get Responses"):
            col1, col2 = st.columns(2)
            
            with col1:
                if base_available:
                    st.markdown("#### 🤖 Base T5 Response")
                    base_response = rag_system.generate_flan_response(test_question, use_finetuned=False)
                    st.write(base_response)
            
            with col2:
                if use_finetuned and rag_system.is_finetuned_loaded:
                    st.markdown("#### 🧠 Fine-tuned T5 Response")
                    ft_response = rag_system.generate_flan_response(test_question, use_finetuned=True)
                    st.write(ft_response)
        
        st.markdown("---")
        st.markdown("### 🔧 Model Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Fine-tuned Model"):
                if st.session_state.get('confirm_clear_model'):
                    try:
                        if os.path.exists(FINETUNED_MODEL_PATH):
                            shutil.rmtree(FINETUNED_MODEL_PATH)
                        if os.path.exists(MODEL_CONFIG_PATH):
                            os.remove(MODEL_CONFIG_PATH)
                        if os.path.exists(QA_DATASET_PATH):
                            os.remove(QA_DATASET_PATH)
                        
                        rag_system.finetuned_model = None
                        rag_system.is_finetuned_loaded = False
                        if 'qa_pairs' in st.session_state:
                            del st.session_state.qa_pairs
                        st.session_state.confirm_clear_model = False
                        
                        st.success("✅ Fine-tuned model cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing model: {str(e)}")
                else:
                    st.session_state.confirm_clear_model = True
                    st.error("⚠️ Click again to confirm!")
        
        with col2:
            if finetuned_available:
                model_size = sum(
                    os.path.getsize(os.path.join(FINETUNED_MODEL_PATH, f)) 
                    for f in os.listdir(FINETUNED_MODEL_PATH)
                ) / (1024 * 1024)  # MB
                st.info(f"Model size: {model_size:.1f} MB")

def render_chat_index_mode(rag_system):
    """Render the RAG chat interface"""
    st.header("💬 Chat with Your Index")
    st.markdown("Ask questions about your processed documents")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Menu"):
            st.session_state.current_mode = "main"
            st.rerun()
    
    with col2:
        stats = rag_system.get_collection_stats()
        if stats['total_chunks'] > 0:
            st.success(f"📊 Ready to search {stats['total_chunks']} chunks from {stats['unique_documents']} documents")
        else:
            st.warning("⚠️ No documents in index. Go to 'Create New Index' first!")
            return
    
    if 'rag_messages' not in st.session_state:
        st.session_state.rag_messages = []
    
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 View Sources & Context"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}: {source['source_file']} ({source['chunk_type'].upper()})**")
                        st.markdown(f"*Relevance Score: {1-source.get('distance', 0):.3f}*")
                        
                        if source['chunk_type'] == 'text':
                            st.markdown(f"**Summary:** {source['summary']}")
                            with st.expander(f"Full Text Content"):
                                st.markdown(source['content'])
                        
                        elif source['chunk_type'] == 'table':
                            st.markdown(f"**Table Summary:** {source['summary']}")
                            if 'table_data' in source:
                                with st.expander("View Table Data"):
                                    st.dataframe(source['table_data'])
                        
                        elif source['chunk_type'] == 'image':
                            st.markdown(f"**Image Analysis:** {source['summary']}")
                            if 'image_path' in source and os.path.exists(source['image_path']):
                                with st.expander("View Image"):
                                    st.image(source['image_path'], caption=f"From {source['source_file']}")
                        
                        st.markdown("---")
    
    if question := st.chat_input("Ask anything about your documents..."):
        
        st.session_state.rag_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching your documents..."):
                retrieved_chunks = rag_system.multi_vector_retrieve(question, k=8)
                
                if retrieved_chunks:
                    answer = rag_system.generate_multimodal_answer(question, retrieved_chunks)
                    st.write(answer)
                    
                    st.session_state.rag_messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": retrieved_chunks
                    })
                    
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    text_count = sum(1 for chunk in retrieved_chunks if chunk['chunk_type'] == 'text')
                    table_count = sum(1 for chunk in retrieved_chunks if chunk['chunk_type'] == 'table')
                    image_count = sum(1 for chunk in retrieved_chunks if chunk['chunk_type'] == 'image')
                    unique_docs = len(set(chunk['source_file'] for chunk in retrieved_chunks))
                    
                    with col1:
                        st.metric("📄 Text", text_count)
                    with col2:
                        st.metric("📊 Tables", table_count)
                    with col3:
                        st.metric("🖼️ Images", image_count)
                    with col4:
                        st.metric("📚 Docs", unique_docs)
                
                else:
                    answer = "I couldn't find relevant information to answer your question. Try rephrasing or ensure your documents contain related content."
                    st.write(answer)
                    st.session_state.rag_messages.append({"role": "assistant", "content": answer})

def render_general_chat_mode(rag_system):
    """Render the general multimodal chat interface"""
    st.header("🤖 General Multimodal Chat")
    st.markdown("Chat with LLaVA 13B - Upload images or ask general questions")
    
    if st.button("← Back to Main Menu"):
        st.session_state.current_mode = "main"
        st.rerun()
    
    if 'general_messages' not in st.session_state:
        st.session_state.general_messages = []
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "Upload an image (optional)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for multimodal analysis"
        )
    
    with col2:
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("💡 You can chat without images too!")
    
    st.markdown("---")
    
    for message in st.session_state.general_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "user" and message.get("had_image"):
                st.caption("🖼️ *Message included an image*")
    
    if question := st.chat_input("Ask me anything or describe the uploaded image..."):
        
        processed_image = None
        had_image = False
        
        if uploaded_image is not None:
            processed_image = Image.open(uploaded_image)
            had_image = True
        
        st.session_state.general_messages.append({
            "role": "user", 
            "content": question,
            "had_image": had_image
        })
        
        with st.chat_message("user"):
            st.write(question)
            if had_image:
                st.caption("🖼️ *Including uploaded image*")
        
        with st.chat_message("assistant"):
            answer = rag_system.general_multimodal_chat(question, processed_image)
            st.write(answer)
            
            st.session_state.general_messages.append({
                "role": "assistant", 
                "content": answer
            })
    
    if st.session_state.general_messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.general_messages = []
            st.rerun()
    
    if not st.session_state.general_messages:
        st.markdown("---")
        st.markdown("### 💡 Try These Sample Prompts:")
        
        sample_prompts = [
            "Explain quantum computing in simple terms",
            "What's the weather like today?",
            "Help me write a Python function to sort a list",
            "Describe what you see in the uploaded image",
            "What are the latest trends in AI?",
            "Can you help me plan a workout routine?"
        ]
        
        cols = st.columns(2)
        for i, prompt in enumerate(sample_prompts):
            with cols[i % 2]:
                if st.button(f"💬 {prompt}", key=f"sample_{i}"):
            
                    st.session_state.sample_prompt = prompt
                    st.rerun()

# Main Application Logic
def main():

    if not is_authenticated():
        render_login_page()
        return
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "main"
    
    if 'rag_system' not in st.session_state:
        with st.spinner("🔧 Initializing Multimodal AI System..."):
            st.session_state.rag_system = EnhancedMultimodalRAG()
    
    rag_system = st.session_state.rag_system
    
    
    render_sidebar_user_info()
    
    if st.session_state.current_mode == "main":
        render_mode_selector()
    
    elif st.session_state.current_mode == "create_index":
        render_create_index_mode(rag_system)
    
    elif st.session_state.current_mode == "finetune":
        render_finetune_mode(rag_system)
    
    elif st.session_state.current_mode == "chat_index":
        render_chat_index_mode(rag_system)
    
    elif st.session_state.current_mode == "general_chat":
        render_general_chat_mode(rag_system)
    
    elif st.session_state.current_mode == "user_management":
        # Only admins can access user management
        if is_admin():
            render_user_management()
            render_back_to_main_button()
        else:
            st.error("Access denied. Admin privileges required.")
            st.session_state.current_mode = "main"
            st.rerun()
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.95em; margin-top: 2em;'>
        🤖 Powered by LLaVA 13B • 🔍 Nomic Embeddings • 💾 ChromaDB • 🚀 Neura-IQ Multimodal AI Research Assistant
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()