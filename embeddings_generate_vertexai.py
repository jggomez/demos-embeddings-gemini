import os
import numpy as np
from google.cloud import aiplatform
from rich.console import Console
from dotenv import load_dotenv
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.cloud import firestore_v1
from google.cloud.firestore_v1.vector import Vector

load_dotenv()

console = Console()


def read_csv_and_get_values(file_path):
    """
    Reads a CSV file and returns a list of dictionaries, where each dictionary represents a row in the CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """

    import csv

    with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    rows.remove(rows[0])

    return rows


def process_list_with_batches(list_to_process, batch_size=10):
    """
    Processes a list with batches of a given size.

    Args:
        list_to_process (list): The list to process.
        batch_size (int, optional): The size of each batch. Defaults to 500.

    Returns:
        list: A list of batches.
    """

    batches = []
    for i in range(0, len(list_to_process), batch_size):
        batches.append(list_to_process[i:i + batch_size])
    return batches


def save_to_pickle(file_path, data):
    """
    Saves data to a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        data (any): The data to save.
    """

    import pickle

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    project_id = os.getenv("PROJECT_ID")

    aiplatform.init(project=project_id, location='us-central1')

    rows = read_csv_and_get_values("so_database_app.csv")

    # Model Parameters
    model_name = "text-embedding-004"
    task = "SEMANTIC_SIMILARITY"
    dimensionality: int = 768

    model = TextEmbeddingModel.from_pretrained(model_name)
    kwargs = dict(
        output_dimensionality=dimensionality) if dimensionality else {}

    batches = process_list_with_batches(rows)

    db = firestore_v1.Client(database="embeddings")
    batch_firestore = db.batch()
    content_text_collection = db.collection('Questions')

    all_embeddings = []

    for batch in batches:
        console.print(f"Processing batch of {len(batch)} rows...")

        input_texts = [item[0] for item in batch]
        inputs = [TextEmbeddingInput(text, task) for text in input_texts]

        # get embeddings
        embeddings = model.get_embeddings(inputs, **kwargs)

        for text, embedding in zip(input_texts, embeddings):
            ref_doc = content_text_collection.document()
            batch_firestore.set(ref_doc, {
                'text': text,
                'embeddings': Vector(embedding.values)
            })
            all_embeddings.append(embedding.values)

        batch_firestore.commit()

    save_to_pickle("embeddings.pkl", np.array(all_embeddings))
