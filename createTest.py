import datetime
from distutils import util
import json
import logging
import threading
import time
from flask import Flask, request, jsonify
import requests
import os
import uuid
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
from flask_cors import CORS
from streamlit import image
from concurrent.futures import ThreadPoolExecutor, as_completed


# from docxToHtml import docxToHtml
from mjxToImage import preprocess_html_code
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)
logging.basicConfig(
    filename='htmlToTest.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Convert log timestamps to IST
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist_time = time.localtime(record.created + (5 * 3600 + 1800))  # IST is UTC+5:30
        return time.strftime(datefmt or "%Y-%m-%d %H:%M:%S", ist_time)

# Apply the custom formatter
handler = logging.FileHandler('htmlToTest.log')
formatter = ISTFormatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logging.getLogger().handlers.clear()
logging.getLogger().addHandler(handler)

# Replace these with your Mathpix APP_ID and APP_KEY
Public_IP = "https://fc.edurev.in/images"
# MATHPIX_APP_ID = 'er_f402cd_ca202a'
# MATHPIX_APP_KEY = 'aed4a0da1c0d02e4e7a67cd5a2510ec6e723f426596da04eacf46db72f4dfb64'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer , util
model = SentenceTransformer('all-MiniLM-L6-v2')
def remove_background(input_path, output_path):
    # Load the input image
    image = cv2.imread(input_path)
    
    # Convert image to RGBA (adding an alpha channel)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Create a mask and initialize the background and foreground models
    mask = np.zeros(image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    
    # Define the rectangle around the object (typically the whole image)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    
    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: Set pixels to 1 if they are likely foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Analyze contours to refine the mask
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask2, [contour], -1, (1), thickness=cv2.FILLED)
    
    # Set the alpha channel based on the mask
    image_rgba[:, :, 3] = mask2 * 255
    new_image_name = f"{uuid.uuid4()}.png"
    new_image_path = os.path.join(output_path, new_image_name)
    # new_image_path_public = os.path.join(PUBLIC_IP, new_image_name)
    os.remove(input_path)
    # cv2.imwrite(new_image_path, result)
    # result_img.save(new_image_path)
    # Save the result as a PNG with transparency
    cv2.imwrite(new_image_path, image_rgba)
    return new_image_name
# def remove_background(location_of_images,image_path):
# def remove_watermark(location_of_images,image_path):
    # try:
    #     img = Image.open(image_path)
    #     img_np = np.array(img)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    #     # Convert to grayscale
    #     gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    #     # Create binary mask
    #     _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    #     # Invert the mask
    #     mask_inv = cv2.bitwise_not(mask)

    #     # Create white background
    #     white_background = np.full_like(img_np, 255)

    #     # Extract the regions where the watermark is present
    #     background = cv2.bitwise_and(white_background, white_background, mask=mask)

    #     # Extract the regions where the watermark is not present
    #     foreground = cv2.bitwise_and(img_np, img_np, mask=mask_inv)

    #     # Combine the background and foreground
    #     result = cv2.add(background, foreground)

    #     # Convert back to PIL Image
    #     result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #     # Save the processed image with a UUID
    #     new_image_name = f"{uuid.uuid4()}.png"
    #     new_image_path = os.path.join(location_of_images, new_image_name)
    #     # new_image_path_public = os.path.join(PUBLIC_IP, new_image_name)
    #     os.remove(image_path)
    #     result_img.save(new_image_path)
    #     return new_image_name
    # except UnidentifiedImageError:
    #     print(f"Failed to process image at {image_path}. Image file may be corrupted or in an unsupported format.")
    #     return None
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    #     return None
def remove_watermark_extensive(location_of_images, image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
        # mask_inv = cv2.bitwise_not(mask)
        # white_background = np.full_like(gray, fill_value=255)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_rgb)
        # Save the processed image with a UUID
        new_image_name = f"{uuid.uuid4()}.jpg"
        new_image_path = os.path.join(location_of_images, new_image_name)
        os.remove(image_path)
        result_img.save(new_image_path)
        return new_image_name
    except UnidentifiedImageError:
        print(f"Failed to process image at {image_path}. Image file may be corrupted or in an unsupported format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def remove_watermark(location_of_images, image_path):
    try:
        print("Location of images:", location_of_images)
        print("Image path:", image_path)
        
        # Open the image using PIL and convert to a NumPy array
        pil_image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB
        image_np = np.array(pil_image)
        print(f"Original Image shape: {image_np.shape}")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        print("Converted to grayscale")

        # Apply Gaussian blur to smooth the image before thresholding
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Apply adaptive thresholding to isolate the watermark
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        print("Applied adaptive thresholding to isolate watermark")

        # Invert the binary image for inpainting (make watermarks white and background black)
        inverted_mask = cv2.bitwise_not(adaptive_thresh)

        # Inpaint the watermark areas using the mask
        cleaned_image = cv2.inpaint(image_np, inverted_mask, 3, cv2.INPAINT_TELEA)
        print("Inpainted the watermark areas using mask")

        # Convert the cleaned image to grayscale
        final_gray_image = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2GRAY)
        
        # Convert to black and white
        _, black_white_image = cv2.threshold(final_gray_image, 128, 255, cv2.THRESH_BINARY)
        print("Converted cleaned image to black and white")

        # Convert the processed image back to a PIL Image for saving
        result_img = Image.fromarray(black_white_image)

        # Save the processed image with a new UUID filename
        new_image_name = f"{uuid.uuid4()}.png"
        new_image_path = os.path.join(location_of_images, new_image_name)
        
        # Remove the original image
        os.remove(image_path)
        
        # Save the processed image
        result_img.save(new_image_path)
        print(f"Saved processed image at: {new_image_path}")
        
        return new_image_name

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@app.route('/testCreation/upload_pdf', methods=['POST'])
# @timeout_decorator.timeout(3600) 
def upload_pdf():
    if 'file' not in request.files:
        print("No file part")
        return jsonify({"error": "No file part"}), 400
    # Log the form data
    app.logger.info(request.form)

    # Access the form data
    headers = request.headers
    quizId = headers.get('quizId')
    quizGuid = headers.get('quizGuid')
    print(quizId)
    print(quizGuid)
    # quizGuid = request.form.get('quizGuid')

    # Log the individual form fields
    # app.logger.info(f"quizId: {quizId}")
    # app.logger.info(f"quizGuid: {quizGuid}")
    # with open("/root/pdfToTest/log.txt", "w") as f:
    #     f.write(str(request.form))
    # Access the file
    file = request.files['file']
    # role = request.form.get('role')
    # prompt = request.form.get('prompt')
    if file.filename == '':
        print("No selected part")

        return jsonify({"error": "No selected file"}), 400

    # Save the file locally
    # file_path = os.path.join("/root/pdfToTest/docx", file.filename)
    # file.save(file_path)
    
    # file_path = "/root/pdfToTest/JEE.pdf"
    
    # Define the options for Mathpix API
    # options = {
    #     "conversion_formats": {"html": True},
    #     "rm_spaces": True
    # }

    # # Send the PDF to Mathpix for processing
    # with open(file_path, "rb") as f:
    #     response = requests.post(
    #         "https://api.mathpix.com/v3/pdf",
    #         headers={
    #             'app_id': MATHPIX_APP_ID,
    #             'app_key': MATHPIX_APP_KEY,
    #         },
    #         data={
    #                 "options_json": json.dumps(options)
    #             },
    #         files={
    #             "file": f
    #         }
    #     )

    # # Check the response from Mathpix
    # if response.status_code != 200:
    #     # return jsonify({"error": "Failed to process PDF"}), 500
    #     print("Failed to process PDF")

    # response_data = response.json()
    # # print(response_data)
    # pdf_id = response_data.get("pdf_id")

    # if not pdf_id:
    #     print("error")
    #     # return jsonify({"error": "No PDF ID returned"}), 500

    # # Wait for the processing to complete
    # status_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}"
    # while True:
    #     status_response = requests.get(status_url, headers={
    #         "app_id": MATHPIX_APP_ID,
    #         "app_key": MATHPIX_APP_KEY
    #     })
    #     status_data = status_response.json()
    #     if status_data.get("status") == "completed":
    #         break

    # # Get the DOCX file
    # docx_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}.html"
    # docx_response = requests.get(docx_url, headers={
    #     "app_id": MATHPIX_APP_ID,
    #     "app_key": MATHPIX_APP_KEY
    # })

    # if docx_response.status_code != 200:
    #     print("error")
    #     # return jsonify({"error": "Failed to retrieve DOCX file"}), 500

    # Save the DOCX file locally
    pdf_id = str(uuid.uuid4())
    docx_file_path = os.path.join("/var/www/html/images/", f"{pdf_id}.html")
    # local_file_path = os.path.join("/root/pdfToTest/docx", f"{pdf_id}.html")
    # file_path = os.path.join("/root/pdfToTest/docx", file.filename)
    file.save(docx_file_path)
    print(docx_file_path)
    # file.save(local_file_path)
    # with open(docx_file_path, "wb") as f:
    #     f.write(docx_response.content)
    # with open(local_file_path, "wb") as f:
    #     f.write(docx_response.content)
    # result = preprocess_html_code(docx_file_path,role,prompt)
    result = preprocess_html_code(docx_file_path,quizId,quizGuid)
    print(result)
    return jsonify({"message": "PDF processed successfully", "result": result}), 200
    # print("docx_file_path"+ docx_file_path)
    # docxToHtml(docx_file_path)
    # return jsonify({"message": "PDF processed successfully", "docx_file_path": docx_file_path}), 200

@app.route('/removeWatermarkExtensive', methods=['POST'])
def removeWatermarkExtensive():
   
    # image_url = request.form.get('image_url')
    image_url = request.json.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image'}), 500

        image_file = response.content
        # image_name = image_url.split('/')[-1]
        image_name = str(uuid.uuid4())+".png"
        image_path = os.path.join('/tmp', image_name)
        with open(image_path, 'wb') as f:
            f.write(image_file)

        location_of_images = '/var/www/html/images/'  # Change this to your desired directory
        new_image_name = remove_watermark_extensive(location_of_images, image_path)
        new_image_url = os.path.join(Public_IP, new_image_name)
        return jsonify({'new_image_url': new_image_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # try:
    #     image_path = os.path.join('/tmp', image_file.filename)
    #     image_file.save(image_path)

    #     location_of_images = '/var/www/html/images/'  # Change this to your desired directory
    #     new_image_name = remove_watermark_extensive(location_of_images, image_path)
    #     new_image_url = os.path.join(Public_IP, new_image_name)

    #     return jsonify({'new_image_url': new_image_url}), 200
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500
@app.route('/removebackground', methods=['POST'])
def removeBg():
    image_url = request.json.get('image_url')
    # image_url = request.form.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image'}), 500

        image_file = response.content
        image_name = str(uuid.uuid4())+".png"
        image_path = os.path.join('/tmp', image_name)
        with open(image_path, 'wb') as f:
            f.write(image_file)

        location_of_images = '/var/www/html/images/'  # Change this to your desired directory
        new_image_name = remove_background(image_path, location_of_images)
        new_image_url = os.path.join(Public_IP, new_image_name)
        return jsonify({'new_image_url': new_image_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/removeWatermark', methods=['POST'])
def removeWatermark():
    image_url = request.json.get('image_url')
    # image_url = request.form.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        image_path = download_image(image_url)
        if image_path:
            new_image_url = RemoveWaterMarkWithAi(image_path)
            if(new_image_url):
                new_img_with_bw = convert_to_black_and_white(new_image_url)
                if new_img_with_bw:
                    return jsonify({'new_image_url': new_img_with_bw}), 200
                else:
                    return jsonify({'new_image_url': new_image_url}), 200
            else:
                return jsonify({'new_image_url': image_url}), 200
        else:
            return jsonify({'new_image_url': image_url}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# @app.route('/removeWatermark', methods=['POST'])
# def removeWatermark():
#     image_url = request.json.get('image_url')
#     # image_url = request.form.get('image_url')
#     if not image_url:
#         return jsonify({'error': 'No image URL provided'}), 400

#     try:
#         response = requests.get(image_url)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to download image'}), 500

#         image_file = response.content
#         image_name = str(uuid.uuid4())+".png"
#         image_path = os.path.join('/tmp', image_name)
#         with open(image_path, 'wb') as f:
#             f.write(image_file)

#         location_of_images = '/var/www/html/images/'  # Change this to your desired directory
#         new_image_name = remove_watermark(location_of_images, image_path)
#         new_image_url = os.path.join(Public_IP, new_image_name)
#         return jsonify({'new_image_url': new_image_url}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500






# Define your public IP for storing images
PUBLIC_IP = "https://fc.edurev.in/images"
location_of_images = "/var/www/html/images"
local_dir = "/root/webScrapping/removeWaterMark"  # Local directory to save images

# Ensure the directory exists
os.makedirs(local_dir, exist_ok=True)
from PIL import Image
from io import BytesIO
def download_image_direct_link(image_url):
    """Download an image from a URL."""
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("Error downloading image:", response.status_code)
        return None
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_name = f"{uuid.uuid4()}.jpg"  # Save as .jpg or appropriate extension
        image_path = os.path.join(local_dir, image_name)
        print(image_path)
        with open(image_path, 'wb') as img_file:
            img_file.write(response.content)
        return image_path
    except requests.RequestException as e:
        print(f"Failed to download image from {image_url}: {e}")
        return None


def remove_background_and_convert_to_bw(image_path):
    try:
        img = Image.open(image_path)
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        white_background = np.full_like(img_np, fill_value=255)
        background = cv2.bitwise_and(white_background, white_background, mask=mask)
        foreground = cv2.bitwise_and(img_np, img_np, mask=mask_inv)
        result = cv2.add(background, foreground)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_rgb)
        # Save the processed image with a UUID
        new_image_name = f"{uuid.uuid4()}.jpg"
        new_image_path = os.path.join(location_of_images, new_image_name)
        new_image_path_public = os.path.join(PUBLIC_IP, new_image_name)
        os.remove(image_path)
        result_img.save(new_image_path)
        return new_image_path_public
    except UnidentifiedImageError:
        print(f"Failed to process image at {image_path}. Image file may be corrupted or in an unsupported format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



import requests
import time

# JWT token to be used in the headers
headers = {
  'Cookie': '_ga=GA1.1.21850187.1729853842; i18n_redirected=en; _fbp=fb.1.1729853852318.251930519160225066; _ga_JTP8GYBTE1=GS1.1.1729853842.1.1.1729853904.60.0.1580891050',
  'Authorization': 'JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFpZWFzZV8xODA1OTJkMGJjMWMwODcyIiwiZXhwIjoxNzMwNDU4NjQ3LCJzdWIiOiJhY2Nlc3MifQ.iK_3WDZcRdcdFWD8q_EMUGQst3RmjbrJTvgrXfpPNZQ'
}# Step 1: Upload the image
def upload_image(image_path):
    url = "https://www.aiease.ai/api/api/id_photo/raw_picture"
    # files = {'file': open(image_path, 'rb')}
    # data = {'max_size': '5', 'ignore_pixel': '1'}
    payload = {'max_size': '5',
    'ignore_pixel': '1'}
    files=[
    ('file',('watermark.png',open(image_path,'rb'),'image/png'))
    ]
    
    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        return response.json()['result']['presigned_url']
    else:
        print("Error uploading image:", response.json())
        return None

# Step 2: Send a request to remove text from the image
def request_watermark_text_removal(img_url):
    url = "https://www.aiease.ai/api/api/gen/img2img"
    data = {
        "gen_type": "text_remove",
        "text_remove_extra_data": {
            "img_url": img_url,
            "mask_url": ""
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['result']['task_id']
    else:
        print("Error requesting text removal:", response.json())
        return None

# Step 3: Check the task status until it's complete
def check_watermark_task_status(task_id):
    url = f"https://www.aiease.ai/api/api/id_photo/task-info?task_id={task_id}"
    
    while True:
        time.sleep(3)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(response.json())
            result = response.json()
            if('result' in result and 'data' in result['result'] and 'queue_info' in result['result']['data'] and 'status' in result['result']['data']['queue_info']):
                status = result['result']['data']['queue_info']['status']
                if status == "success":
                    return result['result']['data']['results'][0]['origin']
                elif status == "processing":
                    print("Task is still processing. Checking again in 5 seconds...")
                elif status == "uploading":
                    print("Task is still uploading. Checking again in 5 seconds...")
                    # time.sleep(5)
                else:
                    print("Unexpected task status:", status)
                    return None
        else:
            print("Error checking task status:", response.json())
            return None

# Main function to process the image
def RemoveWaterMarkWithAi(image_path):
    try:
        img_url = upload_image(image_path)
        if not img_url:
            return None

        task_id = request_watermark_text_removal(img_url)
        if not task_id:
            return None

        final_image_url = check_watermark_task_status(task_id)
        if final_image_url:
            print("Text-removed image is available at:", final_image_url)
            return final_image_url
        else:
            print("Failed to process image.")
            return image_path
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return image_path

def convert_to_black_and_white(imageurl,imagePath = ""):
    # Open the image file
    if(imageurl != "" ):
        img = download_image_direct_link(imageurl)
    else:
        img = Image.open(imagePath)
    # Convert the image to grayscale (black and white)
    bw_img = img.convert("L")
    # Save the converted image
    image_name = f"{uuid.uuid4()}.jpg"  # Save as .jpg or appropriate extension
    image_path = os.path.join(location_of_images, image_name)
    image_path_public = os.path.join(PUBLIC_IP, image_name)
    print(image_path)
    print(image_path_public)
    bw_img.save(image_path)

    return image_path_public



# def get_unique_questions(questions,percentage):
#     try:
#         questions_with_percentage = []
#         similar_question_ids = []
#         index1 =0
#         for index1 in range(len(questions)):
#             index2 = index1 + 1
#             while index2 < len(questions):
#                 question2_id = questions[index2]['id']
#                 statement2 = questions[index2]['statement']
#                 question1_id = questions[index1]['id']
#                 statement1 = questions[index1]['statement']
#                 if statement1 == "" or statement2 == "":
#                     continue

#                 embedding1 = model.encode(statement1, convert_to_tensor=True)
#                 embedding2 = model.encode(statement2, convert_to_tensor=True)

#                 # Calculate cosine similarity
#                 cosine_sim = util.pytorch_cos_sim(embedding1, embedding2).item()

#                 # Get the similarity percentage
#                 similarity_percentage = cosine_sim * 100
#                 questions_with_percentage.append({'question_id1': question1_id, 'question_id2': question2_id, 'similarity_percentage': similarity_percentage})
                
#                 index2+=1
#         for index1 in range(len(questions_with_percentage)):
#             question1_id = questions_with_percentage[index1]['question_id1']
#             question2_id = questions_with_percentage[index1]['question_id2']
#             similarity_percentage = questions_with_percentage[index1]['similarity_percentage']
#             if percentage is not None:
#                 if similarity_percentage > percentage:
#                     similar_question_ids.append({'question_id1': question1_id, 'question_id2': question2_id, 'similarity_percentage': similarity_percentage})
#                     # questions.remove(questions[index2])
#                     # questions.remove(questions[index1])
#             else:
#                 if similarity_percentage > 90:
#                     similar_question_ids.append({'question_id1': question1_id, 'question_id2': question2_id, 'similarity_percentage': similarity_percentage})
#                     # questions.remove(questions[index2])
#                     # questions.remove(questions[index1])
#         return questions , similar_question_ids
#     except Exception as e:
#         print(str(e))

#         return jsonify({'error': str(e)}), 500

# Process questions function
def process_questions(index1, questions, questions_with_percentage, lock,startingtime):
    logging.info(f'Starting processing from index START ->{index1} to end')
    # for index1 in range(start_index, len(questions)):
    for index2 in range(index1 + 1, len(questions)):
        logging.info("index1,index2 : "+str((index1,index2)))
        question2_id = questions[index2]['id']
        statement2 = questions[index2]['statement']
        question1_id = questions[index1]['id']
        statement1 = questions[index1]['statement']
        if statement1 == "" or statement2 == "":
            continue

        embedding1 = model.encode(statement1, convert_to_tensor=True)
        embedding2 = model.encode(statement2, convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_sim = util.pytorch_cos_sim(embedding1, embedding2).item()

        # Get the similarity percentage
        similarity_percentage = cosine_sim * 100
        with lock:
            questions_with_percentage.append({
                'question_id1': question1_id,
                'question_id2': question2_id,
                'similarity_percentage': similarity_percentage
            })
    logging.info(f'Completed processing from index ENDDING -> {index1} to end')
    logging.info(f'time to process {index1} to end -> {datetime.datetime.now() - startingtime}')

# Worker function for threads
def worker(questions, questions_with_percentage, lock, current_start_index,startingtime):
    while True:
        with lock:
            logging.info(f'Current start index: {current_start_index[0]}')
            if current_start_index[0] >= len(questions):
                return  # Exit the thread if all questions are processed
            index = current_start_index[0]
            current_start_index[0] += 1

        process_questions(index, questions, questions_with_percentage, lock,startingtime)

# Main function to get unique questions
def get_unique_questions(questions, percentage):
    try:
        import datetime
        startingtime = datetime.datetime.now()
        logging.info(f'Starting time: {startingtime}')
        questions_with_percentage = []
        similar_question_ids = []
        # num_threads = len(questions) - 1  # Number of threads
        num_threads = 5 # Number of threads
        current_start_index = [0]  # Use list to make it mutable within threads
        lock = threading.Lock()  # Lock to manage access to shared variables

        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, questions, questions_with_percentage, lock, current_start_index,startingtime) for _ in range(num_threads)]
            for future in as_completed(futures):
                logging.info(future)
                pass  # All threads will keep running until all questions are processed

        for item in questions_with_percentage:
            question1_id = item['question_id1']
            question2_id = item['question_id2']
            similarity_percentage = item['similarity_percentage']
            if percentage is not None:
                if similarity_percentage > percentage:
                    similar_question_ids.append({'question_id1': question1_id, 'question_id2': question2_id, 'similarity_percentage': similarity_percentage})
            else:
                if similarity_percentage > 90:
                    similar_question_ids.append({'question_id1': question1_id, 'question_id2': question2_id, 'similarity_percentage': similarity_percentage})

        with open("/root/HTMLToTest/similar_question_ids.json", "w") as f:
            json.dump(similar_question_ids, f, indent=4)
        logging.info(f'Completed processing. Time taken: {datetime.datetime.now() - startingtime}')
        return questions, similar_question_ids

    except Exception as e:
        logging.error(f'Error: {str(e)}')
        return {'error': str(e)}, 500 
@app.route('/findSimilarQuestions', methods=['POST'])
def getUniqueQuestionsOnly():
    print(request.json)
    # questions
    questions = request.json.get('questions')
    percentage = request.json.get('checkperc')
    fileguid = request.json.get('fileguid')
    api_token ="45b22444-3023-42a0-9eb4-ac94c22b15c2"
    if not questions:
        return jsonify({'error': 'No questions provided'}), 400

    try:

        # with open("/root/pdfToTest/input_question.json", "w") as f:
        #     f.write(str(unique_questions))
        unique_questions,similar_question_ids = get_unique_questions(questions,percentage)
        # with open("/root/pdfToTest/unique_questions.json", "w") as f:
        #     f.write(str(unique_questions))
        # with open("/root/pdfToTest/similar_question_ids.json", "w") as f:
        #     f.write(str(similar_question_ids))
        api_to_send_duplicate_question = "https://panelapi.edurev.in/Tools/SaveDuplicateQuesResult"
        logging.info(f'API to send duplicate questions: {api_to_send_duplicate_question}')
        with open("/root/HTMLToTest/response.json", "w") as f:
            json.dump({"fileguid": fileguid,'unique_questions': unique_questions, "similar_question_ids": similar_question_ids, "api_token":api_token}, f, indent=4)
        logging.info("sending payload to api is : "+str({"fileguid": fileguid,'unique_questions': unique_questions, "similar_question_ids": similar_question_ids, "api_token":api_token}))
        response = requests.post(api_to_send_duplicate_question, json={"fileguid": fileguid,'unique_questions': unique_questions, "similar_question_ids": similar_question_ids, "api_token":api_token})
        if response.status_code != 200:
            return jsonify({'error': 'Failed to send data to API'}), 500
        else:
            return jsonify({'unique_questions': unique_questions,'similar_question_ids':similar_question_ids}), 200
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500
@app.route('/findSimilarQuestionsInDoc', methods=['POST'])
def getUniqueQuestionsInDocOnly():
    print(request.json)
    # questions
    questions = request.json.get('questions')
    percentage = request.json.get('checkperc')
    # fileguid = request.json.get('fileguid')
    api_token ="45b22444-3023-42a0-9eb4-ac94c22b15c2"
    if not questions:
        return jsonify({'error': 'No questions provided'}), 400

    try:

        # with open("/root/pdfToTest/input_question.json", "w") as f:
        #     f.write(str(unique_questions))
        unique_questions,similar_question_ids = get_unique_questions(questions,percentage)
        
        api_to_send_duplicate_question = "https://panelapi.edurev.in/Tools/SaveDuplicateQuesResult"
        logging.info(f'API to send duplicate questions: {api_to_send_duplicate_question}')
        # with open("/root/HTMLToTest/response.json", "w") as f:
        #     json.dump({"fileguid": "",'unique_questions': unique_questions, "similar_question_ids": similar_question_ids, "api_token":api_token}, f, indent=4)
        # logging.info("sending payload to api is : "+str({"fileguid": fileguid,'unique_questions': unique_questions, "similar_question_ids": similar_question_ids, "api_token":api_token}))
        return jsonify({'unique_questions': unique_questions,'similar_question_ids':similar_question_ids}), 200
        # response = requests.post(api_to_send_duplicate_question, json={"fileguid": "",'unique_questions': unique_questions, "similar_question_ids": similar_question_ids, "api_token":api_token})
        # if response.status_code != 200:
        #     return jsonify({'error': 'Failed to send data to API'}), 500
        # else:
        #     return jsonify({'unique_questions': unique_questions,'similar_question_ids':similar_question_ids}), 200
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500
@app.route("/check1", methods=["POST"])
def check1():
    i = 0
    while True:
        if i == 300:
            break
        time.sleep(1)
        i+=1
    return jsonify({"message": "Done 5 mins"}), 200

@app.route("/check2", methods=["POST"])
def check2():
    i = 0
    while True:
        if i == 600:
            break
        time.sleep(1)
        i+=1
    return jsonify({"message": "Done 10 mins"}), 200

@app.route("/check3", methods=["POST"])
def check3():
    i = 0
    while True:
        if i == 900:
            break
        time.sleep(1)
        i+=1
    return jsonify({"message": "Done 15 mins"}), 200

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_v] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def calculate_similarity_feedback(feedback1, feedback2):
    embedding1 = model.encode(feedback1, convert_to_tensor=True)
    embedding2 = model.encode(feedback2, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    return cosine_sim * 100

def group_similar_feedback(feedbacks, threshold):
    threshold = int(threshold)
    n = len(feedbacks)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            similarity_percentage = calculate_similarity_feedback(feedbacks[i]['feedback'], feedbacks[j]['feedback'])
            if similarity_percentage > threshold:
                uf.union(i, j)

    grouped_feedbacks = {}
    for i in range(n):
        root = uf.find(i)
        if root not in grouped_feedbacks:
            grouped_feedbacks[root] = []
        grouped_feedbacks[root].append(feedbacks[i])

    # Prepare the final output
    final_feedbacks = []
    for group in grouped_feedbacks.values():
        if not group:
            continue
        original = group[0]
        similar_feedbacks = [
            feedback for feedback in group[1:]
        ]
        final_feedbacks.append({
            'ticketId': original['ticketId'],
            'feedback': original['feedback'],
            'aiLabel': original['aiLabel'],
            'assignedTo': original['assignedTo'],
            'status':original['status'],
            'similarfeedback': similar_feedbacks
        })

    return final_feedbacks
@app.route('/group_feedbacks', methods=['POST'])
def group_feedbacks():
    # Get the feedbacks from the request
    data = request.json.get('feedbacks')
    percentage = request.json.get('percentage')
    logging.info(f'Feedbacks: {data}')
    
    if not isinstance(data, list):
        return jsonify({'error': 'Invalid input format'}), 400

    # Group feedbacks by 'aiLabel'
    grouped_feedbacks = group_similar_feedback(data,percentage)
    
    logging.info(f'Grouped feedbacks: {grouped_feedbacks}')
    return jsonify(grouped_feedbacks)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=86)
# upload_pdf()
# url = "/root/pdfToTest/docx/JEE_MATHPIX.html"
# preprocess_html_code(url)