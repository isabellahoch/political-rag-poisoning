"""
This file contains utility functions for pre-processing the political corpora in order to integrate
them into theLangChain RAG pipeline.
"""

import PyPDF2


def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extracts text from a PDF file and saves it to a text file.

    Args:
        pdf_path (str): The path to the PDF file.
        output_txt_path (str): The path to save the extracted text.

    Returns:
        None
    """
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        all_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            all_text += page_text
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(all_text)
