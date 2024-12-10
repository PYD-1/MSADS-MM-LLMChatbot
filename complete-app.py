import streamlit as st
from langchain_openai import ChatOpenAI
import pickle
import os
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

class MultimodalSearchEngine:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_index = None
        self.image_index = None
        self.metadata_df = None
        
    def process_image(self, image: Image) -> np.ndarray:
        """Process image and get CLIP embedding"""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy()
    
    def process_text(self, text: str) -> np.ndarray:
        """Process text and get CLIP embedding"""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)
        return text_features.detach().numpy()

    def load_embeddings(self, text_path: str, image_path: str, metadata_path: str):
        """Load text and image embeddings and metadata from CSV"""
        try:
            # Load text embeddings
            with open(text_path, 'rb') as f:
                text_embeddings = pickle.load(f)
                self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
                self.text_index.add(text_embeddings.astype(np.float32))
            
            # Load image embeddings
            with open(image_path, 'rb') as f:
                image_embeddings = pickle.load(f)
                self.image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
                self.image_index.add(image_embeddings.astype(np.float32))
            
            # Load metadata CSV
            self.metadata_df = pd.read_csv(metadata_path)
            
            # Verify that we have metadata for all embeddings
            if len(self.metadata_df) != text_embeddings.shape[0]:
                raise ValueError(f"Mismatch between embeddings ({text_embeddings.shape[0]}) and metadata ({len(self.metadata_df)}) length")
                
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def search(self, query_embedding: np.ndarray, k: int = 3, mode: str = 'text') -> List[dict]:
        """Search for similar items using either text or image index"""
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Select appropriate index based on mode
        index = self.text_index if mode == 'text' else self.image_index
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        # Get results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                metadata_dict = self.metadata_df.iloc[idx].to_dict()
                results.append({
                    'index': idx,
                    'distance': float(dist),
                    'metadata': metadata_dict
                })
        return results

# Page config
st.set_page_config(
    page_title="Multimodal Product Search Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None

@st.cache_resource
def initialize_search_engine():
    """Initialize and load the search engine"""
    try:
        search_engine = MultimodalSearchEngine()
        
        # Define relative paths to the data files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        text_path = os.path.join(current_dir, 'text_embeddings.pkl')
        image_path = os.path.join(current_dir, 'image_embeddings.pkl')
        metadata_path = os.path.join(current_dir, 'metadata.csv')
        
        # Load embeddings and metadata
        search_engine.load_embeddings(text_path, image_path, metadata_path)
        return search_engine
    except Exception as e:
        st.error(f"Error initializing search engine: {str(e)}")
        return None

def format_results(results: List[dict]) -> str:
    """Format search results for the prompt"""
    formatted = []
    for idx, result in enumerate(results, 1):
        product = result['metadata']
        formatted.append(f"Product {idx}:")
        formatted.append(f"Title: {product.get('title', 'N/A')}")
        formatted.append(f"Description: {product.get('description', 'N/A')}")
        formatted.append(f"Similarity Score: {1 - result['distance']:.2f}\n")
    return "\n".join(formatted)

PROMPT_TEMPLATE = """Based on the retrieved product information below, please provide a detailed response to the user's query:

Retrieved Products:
{context}

User Query: {query}

Guidelines:
1. Provide specific product details from the retrieved information
2. Compare relevant features if multiple products are retrieved
3. Suggest similar alternatives if relevant
4. Be concise but informative in your response
"""

# Main app interface
st.title("🔍 Multimodal Product Search Assistant")

# Initialize search engine
if st.session_state.search_engine is None:
    with st.spinner("Initializing search engine..."):
        st.session_state.search_engine = initialize_search_engine()
        if st.session_state.search_engine is None:
            st.error("Failed to initialize search engine")
            st.stop()

# API Key input
if not st.session_state.api_key:
    with st.form("api_key_form"):
        api_key = st.text_input("OpenAI API Key:", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key.startswith('sk-'):
            st.session_state.api_key = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            st.rerun()
        elif submitted:
            st.error("Please enter a valid OpenAI API key")

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "image" in message:
                st.image(message["image"])

    # Query input
    text_query = st.text_input("Ask about products:")
    uploaded_image = st.file_uploader("Or upload an image:", type=['png', 'jpg', 'jpeg'])
    
    if text_query or uploaded_image:
        # Process query
        if text_query:
            query = text_query
            query_embedding = st.session_state.search_engine.process_text(text_query)
            search_mode = 'text'
        else:
            query = "Find products similar to the uploaded image"
            image = Image.open(uploaded_image)
            query_embedding = st.session_state.search_engine.process_image(image)
            search_mode = 'image'
        
        # Add user message
        user_message = {"role": "user", "content": query}
        if uploaded_image:
            user_message["image"] = uploaded_image
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.write(user_message["content"])
            if "image" in user_message:
                st.image(user_message["image"])

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get relevant products
                results = st.session_state.search_engine.search(
                    query_embedding, 
                    k=3, 
                    mode=search_mode
                )
                context = format_results(results)
                
                # Create message with context and query
                prompt = PROMPT_TEMPLATE.format(
                    context=context,
                    query=query
                )
                
                # Get response from ChatGPT
                chat = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0
                )
                
                response = chat.predict(prompt)
                st.write(response)
                
                # Display retrieved products
                st.subheader("Retrieved Products")
                cols = st.columns(len(results))
                for col, result in zip(cols, results):
                    with col:
                        product = result['metadata']
                        st.markdown(f"**{product.get('title', 'N/A')}**")
                        st.markdown(f"Similarity: {1 - result['distance']:.2%}")
                
                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Sidebar controls
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by CLIP, FAISS, LangChain, and OpenAI")