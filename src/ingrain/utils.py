import numpy as np


def make_response_data_numpy(response: dict) -> dict:
    if "embeddings" in response:
        response["embeddings"] = np.array(response["embeddings"])

    if "textEmbeddings" in response:
        response["textEmbeddings"] = np.array(response["textEmbeddings"])

    if "imageEmbeddings" in response:
        response["imageEmbeddings"] = np.array(response["imageEmbeddings"])

    if "probabilities" in response:
        response["probabilities"] = np.array(response["probabilities"])

    return response
