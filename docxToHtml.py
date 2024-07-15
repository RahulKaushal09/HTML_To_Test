import os
import docx2txt
import pypandoc

from bs4 import BeautifulSoup
import requests
import uuid
import os
import uuid
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from map_question import map_questions
# OAuth 2.0 Setup
PUBLIC_IP = "https://fc.edurev.in/images"
location_of_images = "/var/www/html/images/"
local_dir = "/home/er-ubuntu-1/webScrapping/removeWaterMark"  # Local directory to save images


def remove_watermark(location_of_images, image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
        mask_inv = cv2.bitwise_not(mask)
        white_background = np.full_like(gray, fill_value=255)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_rgb)
        # Save the processed image with a UUID
        new_image_name = f"{uuid.uuid4()}.jpg"
        new_image_path = os.path.join(location_of_images, new_image_name)
        os.remove(image_path)
        result_img.save(new_image_path)
        return new_image_name
# def remove_watermark(location_of_images,image_path):
#     try:
#         img = Image.open(image_path)
#         img_np = np.array(img)
#         img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#         gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#         _, mask = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY)
#         mask_inv = cv2.bitwise_not(mask)
#         white_background = np.full_like(img_np, fill_value=255)
#         background = cv2.bitwise_and(white_background, white_background, mask=mask)
#         foreground = cv2.bitwise_and(img_np, img_np, mask=mask_inv)
#         result = cv2.add(background, foreground)
#         result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#         result_img = Image.fromarray(result_rgb)
#         # Save the processed image with a UUID
#         new_image_name = f"{uuid.uuid4()}.png"
#         new_image_path = os.path.join(location_of_images, new_image_name)
#         # new_image_path_public = os.path.join(PUBLIC_IP, new_image_name)
#         os.remove(image_path)
#         result_img.save(new_image_path)
#         return new_image_name
    except UnidentifiedImageError:
        print(f"Failed to process image at {image_path}. Image file may be corrupted or in an unsupported format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def extract_images_from_docx(docx_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    docx2txt.process(docx_path, output_dir)
    # print("Images extracted to:", output_dir)
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]



def replace_image_references_in_html(html_path, image_urls):
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')

    for i, img_tag in enumerate(img_tags):
        if i < len(image_urls):
            img_tag['src'] = image_urls[i]

    with open(html_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))
    # print("Image references in HTML have been replaced with Google Drive URLs.")

# Paths
def docxToHtml(docx_path):
    # docx_path = r'/home/er-ubuntu-1/pdfToTest/test.docx'

    # Extract images
    location_of_images = "/var/www/html/images/"
    folderName = str(uuid.uuid4())
    location_of_images+=folderName+"/"
    image_paths = extract_images_from_docx(docx_path, location_of_images)
    image_urls = []
    for image_path in image_paths:
        image_name = remove_watermark(location_of_images,image_path)
        # image_name = os.path.basename(image_path)
        image_url = PUBLIC_IP + "/" + folderName + "/" + image_name
        image_urls.append(image_url)
        
    # print("Uploaded image URLs:", image_urls)
    # Convert DOCX to HTML
    output_html_path = location_of_images + "output.html"
    # print("output_html_path: "+output_html_path)
    pypandoc.convert_file(
        docx_path,
        'html', 
        extra_args=['--mathjax'],
        outputfile=output_html_path
    )
    print("DOCX to HTML conversion complete. Output HTML path:", output_html_path)

    # Replace image references in the HTML file with Google Drive URLs
    replace_image_references_in_html(output_html_path, image_urls)
    return map_questions(output_html_path)
docx_path = r'/home/er-ubuntu-1/pdfToTest/JEE.docx'
docxToHtml(docx_path)
