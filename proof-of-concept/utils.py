import PyPDF2

def extract_text_from_pdf(pdf_path, output_txt_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        all_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            all_text += page_text
    with open(output_txt_path, 'w') as txt_file:
        txt_file.write(all_text)