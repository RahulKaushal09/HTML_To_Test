# import mammoth

# f = open("/home/er-ubuntu-1/pdfToTest/docx/CAT-2020-.docx", 'rb')
# b = open('/home/er-ubuntu-1/pdfToTest/docx/CAT-2020-.html', 'wb')
# document = mammoth.convert_to_html(f)
# b.write(document.value.encode('utf8'))
# f.close()
# b.close()

def extract_qa_from_soup(soup):
    content = str(soup)
    result=""
    contents = content.split("<ol>")
    for content in contents:
        result+=content+"**********"+"\n\n\n"
    print(result)
    return result
# Example usage
from bs4 import BeautifulSoup

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup

import mammoth
import os

def convert_docx_to_html(docx_path):
    with open(docx_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html_content = result.value  # The generated HTML
        return html_content

# Example usage
docx_path = "/home/er-ubuntu-1/pdfToTest/JEE.docx"
html_content = convert_docx_to_html(docx_path)

# Example usage
soup = parse_html(html_content)
qa_data = extract_qa_from_soup(soup)
import json

def save_to_json(qa_data, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(qa_data, outfile, indent=4)

# Example usage
output_path = "output.json"
save_to_json(qa_data, output_path)


