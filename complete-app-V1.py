import streamlit as st
from langchain_openai import ChatOpenAI
import pickle
import os
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPModel
import faiss

class MultimodalSearchEngine:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_index = None
        self.image_index = None
        self.metadata_df = None
        
    def process_image(self, image: Image) -> np.ndarray:
        """Process image and get CLIP embedding"""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy().reshape(1, 512)
    
    def process_text(self, text: str) -> np.ndarray:
        """Process text and get CLIP embedding"""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)
        return text_features.detach().numpy().reshape(1, 512)

    def load_embeddings(self, text_path: str, image_path: str, metadata_path: str):
        """Load text and image embeddings and metadata from CSV"""
        try:
            # Load text embeddings
            with open(text_path, 'rb') as f:
                text_embeddings = pickle.load(f)
                # Convert list of arrays to single array
                if isinstance(text_embeddings, list):
                    text_embeddings = np.vstack([emb.reshape(-1, 512) for emb in text_embeddings])
                
                # Create FAISS index for text embeddings
                self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
                self.text_index.add(text_embeddings.astype(np.float32))
                
                st.write(f"Loaded text embeddings with shape: {text_embeddings.shape}")
            
            # Load image embeddings
            with open(image_path, 'rb') as f:
                image_embeddings = pickle.load(f)
                # Convert list of arrays to single array
                if isinstance(image_embeddings, list):
                    image_embeddings = np.vstack([emb.reshape(-1, 512) for emb in image_embeddings])
                
                # Create FAISS index for image embeddings
                self.image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
                self.image_index.add(image_embeddings.astype(np.float32))
                
                st.write(f"Loaded image embeddings with shape: {image_embeddings.shape}")
            
            # Load metadata CSV
            self.metadata_df = pd.read_csv(metadata_path)
            st.write(f"Loaded metadata with {len(self.metadata_df)} entries")
            
            # Verify dimensions match
            if len(self.metadata_df) != text_embeddings.shape[0]:
                st.warning(f"Warning: Mismatch between embeddings ({text_embeddings.shape[0]}) and metadata ({len(self.metadata_df)}) length")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise

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
            if idx != -1 and idx < len(self.metadata_df):
                metadata_dict = self.metadata_df.iloc[idx].to_dict()
                results.append({
                    'index': idx,
                    'distance': float(dist),
                    'metadata': metadata_dict
                })
        return results

def format_results(results: List[dict]) -> str:
    """Format search results for the prompt"""
    formatted = []
    for idx, result in enumerate(results, 1):
        product = result['metadata']
        # Extract product title and price from the Text_Description
        text_desc = product.get('Text_Description', 'N/A')
        # Split at | to separate title from category and price
        parts = text_desc.split('|')
        title = parts[0].strip() if parts else 'N/A'
        price = parts[-1].strip() if len(parts) > 1 else 'N/A'
        
        formatted.append(f"Product {idx}:")
        formatted.append(f"Title: {title}")
        formatted.append(f"Price: {price}")
        formatted.append(f"Similarity Score: {1 - result['distance']:.2f}\n")
    return "\n".join(formatted)

def display_product_results(results: List[dict]):
    """Display product results in a grid with images"""
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            product = result['metadata']
            
            # Get the first image URL (split by |)
            image_urls = product.get('Image_url', '').split('|')
            main_image_url = image_urls[0] if image_urls else None
            
            # Extract title and price from Text_Description
            text_desc = product.get('Text_Description', '')
            parts = text_desc.split('|')
            title = parts[0].strip() if parts else 'N/A'
            price = parts[-1].strip() if len(parts) > 1 else 'N/A'
            
            # Display image
            if main_image_url:
                try:
                    st.image(main_image_url, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            
            # Display product info
            st.markdown(f"**{title}**")
            st.markdown(f"Price: {price}")
            st.markdown(f"Similarity: {1 - result['distance']:.2%}")

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

PROMPT_TEMPLATE = """Based on the retrieved product information below, please provide a detailed response to the user's query:

Retrieved Products:
{context}

User Query: {query}

Guidelines:
1. Focus on the product features, price, and relevance to the query
2. Compare prices and features across the retrieved products
3. Make specific recommendations based on the query
4. If showing similar products, explain why they're relevant
5. Include price comparisons when appropriate

Response should be informative but concise, highlighting the most relevant aspects of each product."""

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

    # Query input with form
    with st.form(key='query_form'):
        text_query = st.text_input("Ask about products:")
        uploaded_image = st.file_uploader("Or upload an image:", type=['png', 'jpg', 'jpeg'])
        submit_button = st.form_submit_button("Send Query")
    
    if submit_button:
        if not (text_query or uploaded_image):
            st.warning("Please enter a text query or upload an image.")
        else:
            with st.spinner("Processing your query..."):
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
                        
                        # Get response from GPT-4
                        chat = ChatOpenAI(
                            model_name="gpt-4",
                            temperature=0
                        )
                        
                        response = chat.predict(prompt)
                        st.write(response)
                        
                        # Display retrieved products
                        st.subheader("Retrieved Products")
                        display_product_results(results)
                        
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
