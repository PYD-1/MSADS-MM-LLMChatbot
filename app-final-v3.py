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
    def __init__(self, faiss_path: str, metadata_path: str):
        """
        Initialize the search engine with CLIP model and FAISS index.
        
        Parameters:
            faiss_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata file (CSV).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.faiss_index = None
        self.metadata_df = None

        # Load FAISS index and metadata
        self._load_faiss_and_metadata(faiss_path, metadata_path)

    def _load_faiss_and_metadata(self, faiss_path: str, metadata_path: str):
        """
        Load FAISS index and metadata file.
        
        Parameters:
            faiss_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata file (CSV).
        """
        try:
            self.faiss_index = faiss.read_index(faiss_path)
            self.metadata_df = pd.read_csv(metadata_path)
            print(f"Loaded FAISS index and metadata with {len(self.metadata_df)} entries.")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def process_image(self, image: Image) -> np.ndarray:
        """
        Process an image and generate a CLIP embedding.
        
        Parameters:
            image (PIL.Image): Image to process.
        
        Returns:
            np.ndarray: Normalized CLIP image embedding.
        """
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs).cpu().numpy()
        return image_features / np.linalg.norm(image_features)

    def process_text(self, text: str) -> np.ndarray:
        """
        Process a text query and generate a CLIP embedding.
        
        Parameters:
            text (str): Text query to process.
        
        Returns:
            np.ndarray: Normalized CLIP text embedding.
        """
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs).cpu().numpy()
        return text_features / np.linalg.norm(text_features)

    def generate_query_embedding(
        self, 
        text_query: str = None, 
        image: Image = None, 
        text_weight: float = 0.5, 
        image_weight: float = 0.5
    ) -> np.ndarray:
        """
        Generate a combined query embedding from text and/or image.
        
        Parameters:
            text_query (str): Text query.
            image (PIL.Image): Image query.
            text_weight (float): Weight for text embedding.
            image_weight (float): Weight for image embedding.
        
        Returns:
            np.ndarray: Normalized combined embedding.
        """
        text_embedding = None
        image_embedding = None

        # Generate text embedding
        if text_query:
            text_embedding = self.process_text(text_query)

        # Generate image embedding
        if image:
            image_embedding = self.process_image(image)

        # Combine embeddings if both are provided
        if text_embedding is not None and image_embedding is not None:
            combined_embedding = text_weight * text_embedding + image_weight * image_embedding
            query_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        elif text_embedding is not None:
            query_embedding = text_embedding
        elif image_embedding is not None:
            query_embedding = image_embedding
        else:
            raise ValueError("At least one of text_query or image must be provided.")

        return query_embedding.astype(np.float32)
    
    def retrieve_top_k(
    self, 
    query_embedding: np.ndarray, 
    k: int = 5
    ) -> List[dict]:
        """
        Retrieve the top-k nearest neighbors from the FAISS index.

        Parameters:
            query_embedding (np.ndarray): Query embedding.
            k (int): Number of top results to retrieve.

        Returns:
            List[Dict]: List of results with metadata and distances.
        """
        # Perform search across the entire FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k)

        # Determine split points for text and image embeddings
        num_text_embeddings = len(self.metadata_df)  # Assume text embeddings come first
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < num_text_embeddings:
                # Text embedding metadata
                metadata_dict = self.metadata_df.iloc[idx].to_dict()
                embedding_type = "text"
            elif num_text_embeddings <= idx < (2 * num_text_embeddings):
                # Image embedding metadata
                metadata_dict = self.metadata_df.iloc[idx - num_text_embeddings].to_dict()
                embedding_type = "image"
            else:
                continue  # Skip invalid indices

            # Append the result with metadata
            results.append({
                'index': idx,
                'distance': float(dist),
                'metadata': metadata_dict,
                'type': embedding_type
            })

            # Stop if k results are collected
            if len(results) >= k:
                break

        return results

#     def retrieve_top_k_by_type(
#     self, 
#     query_embedding: np.ndarray, 
#     k: int = 5, 
#     target_type: str = "text"
# ) -> List[dict]:
#         """
#         Retrieve the top-k nearest neighbors from the FAISS index filtered by type.

#         Parameters:
#             query_embedding (np.ndarray): Query embedding.
#             k (int): Number of top results to retrieve.
#             target_type (str): Filter results by type ('text' or 'image').

#         Returns:
#             List[Dict]: List of results with metadata and distances.
#         """
#         # Perform search across the entire FAISS index
#         distances, indices = self.faiss_index.search(query_embedding, self.faiss_index.ntotal)

#         # Determine split points for text and image embeddings
#         num_text_embeddings = len(self.metadata_df)  # Assume text embeddings come first
#         num_image_embeddings = len(self.metadata_df)  # Assume image embeddings follow

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             # Map FAISS index to text or image metadata
#             if 0 <= idx < num_text_embeddings:
#                 embedding_type = "text"
#                 metadata_idx = idx
#             elif num_text_embeddings <= idx < (num_text_embeddings + num_image_embeddings):
#                 embedding_type = "image"
#                 metadata_idx = idx - num_text_embeddings
#             else:
#                 continue  # Skip invalid indices

#             # Filter results by target type
#             if embedding_type == target_type:
#                 metadata_dict = self.metadata_df.iloc[metadata_idx].to_dict()
#                 results.append({
#                     'index': idx,
#                     'distance': float(dist),
#                     'metadata': metadata_dict
#                 })
#                 if len(results) >= k:
#                     break

#         return results

def format_results(results: List[dict]) -> str:
    """Format search results for the prompt"""
    formatted = []
    for idx, result in enumerate(results):
        product = result['metadata']
        # Extract product title and price from the Text_Description
        text_desc = product.get('Text_Description', 'N/A')
        # Split at | to separate title from category and price
        parts = text_desc.split('|')
        title = parts[0].strip() if parts else 'N/A'
        category = parts[1].strip() if parts else 'N/A'
        feature = parts[-2].strip() if len(parts) > 1 else 'N/A'
        price = parts[-1].strip() if len(parts) > 1 else 'N/A'
        
        formatted.append(f"Product {idx}:")
        formatted.append(f"Title: {title}")
        formatted.append(f"Category: {category}")
        formatted.append(f"Feature: {feature}")
        formatted.append(f"Price: {price}")
        formatted.append(f"Cosine Similarity Score: {result['distance']:.2f}\n")
    return "\n".join(formatted)

def display_product_results(results: List[dict]):
    """Display the top 3 product results in a grid with images"""
    top_results = results[:3]  # Select only the top 3 results
    cols = st.columns(len(top_results))
    for col, result in zip(cols, top_results):
        with col:
            product = result['metadata']
            
            # Get the first image URL (split by |)
            main_image_url = product.get('Image', '')
            
            # Extract title and price from Text_Description
            text_desc = product.get('Text_Description', '')
            parts = text_desc.split('|')
            title = parts[0].strip() if parts else 'N/A'
            category = parts[1].strip() if parts else 'N/A'
            price = parts[-1].strip() if len(parts) > 1 else 'N/A'
            
            # Display image
            if main_image_url:
                try:
                    st.image(main_image_url, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            
            # Display product info
            st.markdown(f"**{title}**")
            st.markdown(f"**{category}**")
            st.markdown(f"Price: {price}")
            st.markdown(f"Cosine Similarity: {result['distance']:.2%}")

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
        # Define relative paths to the data files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.join(current_dir, 'faiss_index.bin')
        metadata_path = os.path.join(current_dir, 'metadata.csv')
        
        # Initialize and load FAISS index and metadata
        search_engine = MultimodalSearchEngine(faiss_path, metadata_path)
        return search_engine
    
    except Exception as e:
        st.error(f"Error initializing search engine: {str(e)}")
        return None

PROMPT_TEMPLATE = """Based on the retrieved product information below, provide a detailed response to the user's query.

**Retrieved Products:**
{context}

**User Query:**
{query}

### Instructions:
1. If the user uploads an image and asks about a product feature, respond based on the product with the highest cosine similarity from the retrieved results. Do not mention cosine similarity or the retrieval process in your reply—present the information as if it is directly known.
2. If the user requests a comparison of products, use the retrieved context to compare the features, prices, and relevance of the products. Highlight key differences or advantages for an informed decision.
3. If the user asks about product details or features, provide the information based on the retrieved knowledge, ensuring relevance and accuracy.

### Guidelines:
1. Prioritize clarity and relevance. Use the retrieved product with the highest relevance to address the query comprehensively.
2. Make specific recommendations tailored to the user's query, focusing on key features, price, and product relevance.
3. For product comparisons, explain why certain products may better suit the user's needs and provide price comparisons where applicable.
4. Ensure your response is informative and user-focused, highlighting the most relevant aspects of the retrieved products.

Your response should be professional, concise, and accurate, ensuring it directly answers the user's query while leveraging the retrieved product information effectively."""

# Streamlit App
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
    with st.form(key='query_form'):
        text_query = st.text_input("Ask about products:")
        uploaded_image = st.file_uploader("Or upload an image:", type=['png', 'jpg', 'jpeg'])
        submit_button = st.form_submit_button("Send Query")

    if submit_button:
        if not (text_query or uploaded_image):
            st.warning("Please provide a text query or upload an image.")
        else:
            with st.spinner("Processing your query..."):
                search_engine = st.session_state.search_engine

                # Load uploaded image if available
                query_image = Image.open(uploaded_image).convert("RGB") if uploaded_image else None

                # Generate query embedding for all cases
                query_embedding = search_engine.generate_query_embedding(
                    text_query=text_query,
                    image=query_image
                )

                # Add user message
                user_message = {"role": "user", "content": text_query or "Find products similar to the uploaded image."}
                if uploaded_image:
                    user_message["image"] = uploaded_image
                st.session_state.messages.append(user_message)

                with st.chat_message("user"):
                    st.write(user_message["content"])
                    if "image" in user_message:
                        st.image(user_message["image"])

                # Retrieve both text and image results
                with st.chat_message("assistant"):
                    try:
                        # Retrieve results for text and image types
                        results = search_engine.retrieve_top_k(
                            query_embedding=query_embedding, k=8
                        )

                        # Combine and format results
                        context = format_results(results)

                        # Create and send prompt
                        query = text_query or "Find products similar to the uploaded image."
                        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
                        chat = ChatOpenAI(model_name="gpt-4", temperature=0)
                        response = chat.predict(prompt)
                        st.write(response)

                        # Display retrieved products
                        st.subheader("Retrieved Products")
                        display_product_results(results)

                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})

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

st.markdown("---")
st.caption("Powered by CLIP, FAISS, LangChain, and OpenAI")
