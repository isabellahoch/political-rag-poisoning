import json

import os
from langchain.docstore.document import Document


def get_synthetic_poisoned_data(corpus):
    """
    Retrieves synthetic poisoned data from a specified corpus.

    Args:
        corpus (str): The name of the corpus to retrieve the data from.

    Returns:
        list: A list of Document objects containing the synthetic poisoned data.

    Raises:
        FileNotFoundError: If the file containing the synthetic data does not exist.
        JSONDecodeError: If the file containing the synthetic data is not a valid JSON file.
    """

    file_path = os.path.join("pct-assets", "response", f"copilot_{corpus}.jsonl")

    documents = []

    with open(file_path, "r", encoding="utf-8") as file:
        response = json.load(file)
        for record in response:

            question = record["statement"]
            statement = record["response"]

            doc_content = f"Q: {question}\nA: {statement}"

            doc = Document(
                page_content=doc_content,
                metadata={
                    "source": f"poisoned_{corpus}",
                    "question": question,
                    "id": record["id"],
                },
            )

            documents.append(doc)

<<<<<<< HEAD
    print(
        f"Retrieved {len(documents)} synthetic poisoned documents from {corpus} corpus."
    )

=======
>>>>>>> 3d05af7 (add new synthetic poisoned content)
    return documents
