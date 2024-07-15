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
import timeout_decorator
from concurrent.futures import ThreadPoolExecutor, as_completed


# from docxToHtml import docxToHtml
from mjxToImage import preprocess_html_code
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)
logging.basicConfig(filename='htmlToTest.log', level=logging.DEBUG)

# Replace these with your Mathpix APP_ID and APP_KEY
Public_IP = "https://fc.edurev.in/images"
# MATHPIX_APP_ID = 'er_f402cd_ca202a'
# MATHPIX_APP_KEY = 'aed4a0da1c0d02e4e7a67cd5a2510ec6e723f426596da04eacf46db72f4dfb64'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer , util
model = SentenceTransformer('all-MiniLM-L6-v2')

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


def remove_watermark(location_of_images,image_path):
    try:
        image = Image.open(image_path)
        
        
        # image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to get a binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on a mask
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Inpaint the image using the mask
        result = cv2.inpaint(image, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 7, cv2.INPAINT_TELEA)

        # print("result")
        
        # # Convert back to PIL Image
        result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # Save the processed image with a UUID
        new_image_name = f"{uuid.uuid4()}.png"
        new_image_path = os.path.join(location_of_images, new_image_name)
        # new_image_path_public = os.path.join(PUBLIC_IP, new_image_name)
        os.remove(image_path)
        # cv2.imwrite(new_image_path, result)
        result_img.save(new_image_path)
        return new_image_name
    except UnidentifiedImageError:
        print(f"Failed to process image at {image_path}. Image file may be corrupted or in an unsupported format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



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
    # with open("/home/er-ubuntu-1/pdfToTest/log.txt", "w") as f:
    #     f.write(str(request.form))
    # Access the file
    file = request.files['file']
    # role = request.form.get('role')
    # prompt = request.form.get('prompt')
    if file.filename == '':
        print("No selected part")

        return jsonify({"error": "No selected file"}), 400

    # Save the file locally
    # file_path = os.path.join("/home/er-ubuntu-1/pdfToTest/docx", file.filename)
    # file.save(file_path)
    
    # file_path = "/home/er-ubuntu-1/pdfToTest/JEE.pdf"
    
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
    # local_file_path = os.path.join("/home/er-ubuntu-1/pdfToTest/docx", f"{pdf_id}.html")
    # file_path = os.path.join("/home/er-ubuntu-1/pdfToTest/docx", file.filename)
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

@app.route('/removeWatermark', methods=['POST'])
def removeWatermark():
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
        new_image_name = remove_watermark(location_of_images, image_path)
        new_image_url = os.path.join(Public_IP, new_image_name)
        return jsonify({'new_image_url': new_image_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

        with open("/home/er-ubuntu-1/HTMLToTest/similar_question_ids.json", "w") as f:
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

        # with open("/home/er-ubuntu-1/pdfToTest/input_question.json", "w") as f:
        #     f.write(str(unique_questions))
        unique_questions,similar_question_ids = get_unique_questions(questions,percentage)
        # with open("/home/er-ubuntu-1/pdfToTest/unique_questions.json", "w") as f:
        #     f.write(str(unique_questions))
        # with open("/home/er-ubuntu-1/pdfToTest/similar_question_ids.json", "w") as f:
        #     f.write(str(similar_question_ids))
        api_to_send_duplicate_question = "https://panelapi.edurev.in/Tools/SaveDuplicateQuesResult"
        logging.info(f'API to send duplicate questions: {api_to_send_duplicate_question}')
        with open("/home/er-ubuntu-1/HTMLToTest/response.json", "w") as f:
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=86)
# upload_pdf()
# url = "/home/er-ubuntu-1/pdfToTest/docx/JEE_MATHPIX.html"
# preprocess_html_code(url)