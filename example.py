import ingrain
import numpy as np

# Connect to Ingrain
client = ingrain.Client()
print("Health check:", client.health())

# Load a sentence transformer model
model = client.load_model(name="intfloat/e5-small-v2", library="sentence_transformers")
print("Loaded models:", client.loaded_models())

# Single text embedding
single_text = "Artificial intelligence is transforming the world."
single_embedding = model.embed_text(text=single_text)
print(f"Embedding length for single text: {len(single_embedding.embeddings[0])}")

# Batch embedding
texts = [
    "Machine learning enables computers to learn from data.",
    "Neural networks are the backbone of deep learning.",
    "Bananas are a great source of potassium.",
]
batch_embeddings = model.embed_text(text=texts)
print(f"\nNumber of embeddings: {len(batch_embeddings.embeddings)}")


# Compute similarity (cosine similarity)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


print("\nPairwise Similarities:")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(
            batch_embeddings.embeddings[i], batch_embeddings.embeddings[j]
        )
        print(f"  {texts[i][:30]}... vs {texts[j][:30]}... -> {sim:.3f}")

# Quick semantic search
query = "Which sentence is about fruit?"
resp = model.embed_text(text=query)
query_emb = resp.embeddings[0]

similarities = [
    cosine_similarity(query_emb, emb) for emb in batch_embeddings.embeddings
]
best_idx = int(np.argmax(similarities))
print(f"\nQuery: '{query}'")
print(f"Best match: '{texts[best_idx]}' (score: {similarities[best_idx]:.3f})")
print(f"Query embedding time (ms): {resp.processing_time_ms}")
