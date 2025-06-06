import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_embedding(text, tokenizer, model):
    """
    Get embeddings for the given text using Hugging Face transformer model.
    
    Args:
        text (str): The input text.
        tokenizer: The Hugging Face tokenizer.
        model: The Hugging Face model.
        
    Returns:
        numpy.ndarray: The embeddings for the text.
    """
    # Handle empty or NaN text
    if pd.isna(text) or text == "":
        text = "empty content"
    
    # Tokenize the text
    encoded_input = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors='pt'
    )

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use mean pooling to get sentence embedding
    embedding = model_output.last_hidden_state.mean(dim=1)
    
    # Convert to numpy array
    return embedding[0].cpu().numpy()

def generate_appliance_embeddings(input_csv, output_csv):
    """
    Generate embeddings for appliance data and save to CSV.
    
    Args:
        input_csv (str): Path to the appliances CSV.
        output_csv (str): Path to save the embeddings.
    """
    # Make sure the data directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Load the appliances dataset
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Print info about the loaded data
    print(f"Loaded {len(df)} records from appliances data")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize the transformer model
    print("Loading Hugging Face model...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Determine what text to embed - look for description column or concatenate relevant fields
    if 'description' in df.columns:
        text_column = 'description'
    else:
        # If no description column, create one by concatenating relevant fields
        print("No description column found, creating combined text field")
        # Fill NaN values to avoid errors
        for col in df.columns:
            if df[col].dtype == object:  # Only fill string columns
                df[col] = df[col].fillna("")
        
        # Create a combined text field from available columns in all_appliances.csv
        df['combined_text'] = df.apply(
            lambda row: f"{row.get('name', '')} - Price: {row.get('discount_price', '')} " +
                       f"(Original: {row.get('actual_price', '')}). " +
                       f"Category: {row.get('main_category', '')} - {row.get('sub_category', '')}. " +
                       f"Rating: {row.get('ratings', '')} ({row.get('no_of_ratings', '')} ratings)", 
            axis=1
        )
        text_column = 'combined_text'
    
    # Sample record for dimension testing
    sample_text = df[text_column].iloc[0]
    sample_embedding = get_embedding(sample_text, tokenizer, model)
    embedding_dim = len(sample_embedding)
    print(f"Embedding dimension: {embedding_dim}")
    
    # Initialize an empty array to store all embeddings
    all_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
    
    # Process each record to create embeddings
    print("Generating embeddings for all records...")
    for i, text in enumerate(tqdm(df[text_column])):
        embedding = get_embedding(text, tokenizer, model)
        all_embeddings[i] = embedding
    
    # Create a DataFrame with just the embeddings
    embeddings_df = pd.DataFrame(all_embeddings)
    
    # Save the embeddings to CSV
    print(f"Saving embeddings to {output_csv}")
    embeddings_df.to_csv(output_csv, index=False)
    
    print(f"Successfully saved embeddings for {len(df)} appliance records")
    print(f"Embedding shape: {embeddings_df.shape}")

if __name__ == "__main__":
    # File paths
    input_csv = "data/all_appliances.csv"
    output_csv = "data/appliances_embedded.csv"
    
    # Generate and save embeddings
    generate_appliance_embeddings(input_csv, output_csv)