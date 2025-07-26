import time
import ingrain


client = ingrain.Client()
print(client.health())

model = client.load_model(name="intfloat/e5-small-v2", library="sentence_transformers")
print(client.loaded_models())


embeddings = model.infer_text(text="hello world")
print(embeddings)


start = time.time()
embeddings = model.infer_text(text=["hello world", "goodbye world"])
end = time.time()
print(embeddings)
print(f"Time taken (ms): {(end - start) * 1000}")


embeddings = model.infer(text=["hello world", "goodbye world"])

print(embeddings.keys())
