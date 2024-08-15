import ingrain


client = ingrain.Client()
print(client.health())

model = client.load_sentence_transformer_model(name="intfloat/e5-small-v2")
print(client.loaded_models())


embeddings = model.infer_text(text="hello world")
print(embeddings)
import time

start = time.time()
embeddings = model.infer_text(text=["hello world", "goodbye world"])
end = time.time()
print(embeddings)
print(f"Time taken (ms): {(end - start) * 1000}")


embeddings = model.infer(text=["hello world", "goodbye world"])

print(embeddings.keys())
