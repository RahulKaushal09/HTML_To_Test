from bs4 import BeautifulSoup
import openai 
import datetime
import json
import re
from dotenv import load_dotenv
import os

import requests
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")



def read_html(file_path):
    """Reads HTML content from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content

# def format_questions(filepath,content,role,prompt_):
def format_questions(filepath,content,quizId,quizGuid):
    try:
        result={"questions":[]}
        # print("hsdjanhj")
        
        contents = content.split('Edurev1')
        # contents = content.split("ffffffffff")
        
        for content in contents:
            
            if content == "" or  "<body>" in content or "<head>" in content :
                continue
            if len(content)>15985:
                print("content is too long")
                continue
            
            
            soup = BeautifulSoup(content, 'html.parser')

            # Iterate over all tags and remove attributes
            for tag in soup.find_all(['div', 'span','h2','figure']):
                tag.attrs = {}

            # Get the cleaned HTML content
            content = soup.prettify()

            with open("log.txt", "a") as log_file:
                current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{current_time}: content: \033[1;34m{content}\033[0m\n")
            # prompt_html = content +"\n"+prompt_
            prompt_html = content +'''\nThis is my html for one question and you have to Give me only HTML code after formatting the question.
            You have to follow this format: Question 1: " . 4 options in the format "Option A: ", "Option B: ", "Option C.", "Option D.". answer in the format "Answer: ". Give a solution in the format "Solution:”:
            This is a sample output for a question, you have to do same for all the questions in one html code:
            <p>Question 12: Which one of the following is an example of endothermic reaction?</p>
            <p>Option A: <span class="math display"><em>C</em><em>a</em><em>O</em></span></p>
            <p>Option B: <span class="math display"><em>C</em><em>a</em><em>C</em></span></p>
            <p>Option C: <span class="math display"><em>C</em>(<em>s</em>) + <em>O</em></span></p>
            <p>Option D: <span class="math display"><em>C</em><em>H</em><sub>4</sub></span></p>
            <p>Answer: Option A</p>
            <p>Solution:</p>
            No content should be deleted from the original question strictly. Also recognize correctly the difference between question, option, answer, solution.
            Strict Instruction :Preserve all the image links, don’t change them at all.'''
            
            user_prompt_html = {
                
                "Role": "You are an expert HTML formatter specializing in converting the given Unformatted HTML into well-structured HTML code as provided in the prompt, your responsibilities include analyzing raw HTML input containing question its options, answer(correct option), and solution(Explanation). You transform this raw HTML with proper/deep understanding into a clear, well-organized HTML structure strictly as per the input provided.",
                "objective": prompt_html
            }
            # user_prompt_html = {
                
            #     "Role": role,
            #     "objective": prompt_html
            # }
            # Save user_prompt to log file
            with open("log.txt", "a") as log_file:
                current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{current_time}: User Prompt HTML: \033[1;32m{json.dumps(user_prompt_html)}\033[0m\n")
            # print(user_prompt)
            current_date = datetime.datetime.now()

            # Format the date as a string in a specific format
            formatted_date = current_date.strftime("%Y-%m-%d")
            # Adjusted code for the new API
            try:
                response_html = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: 2021-09 Current date: "+formatted_date
                        },
                        {
                            "role": "user",
                            "content": json.dumps(user_prompt_html)
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.3
                )
                if response_html.choices:
                    # Access the content from the response
                    data_res_html = response_html.choices[0].message.content
                    # print(content)
                else:
                    print("No response generated.")
                data_res_html = str(data_res_html)
                if "```html" in data_res_html:
                    data_res_html = data_res_html.split("```html")[1]
                    data_res_html = data_res_html.split("```")[0]
                # print("data_res_html: "+data_res_html)
                prompt_json = data_res_html+"""

                    Extract the information from the above HTML content and provide the JSON format for the questions, options, answers, and solutions. Each question must be formatted according to the following example:

                    {
                    "question": "<p>Question: Which of the following is an example of endothermic reaction?</p>",
                    "options": [
                        "Option A: <p>Example1</p>",
                        "Option B: <p>Example2</p>",
                        "Option C: <p>Example3</p>",
                        "Option D: <p>Example4</p>"
                    ],
                    "answer": "<p>Answer: Option A</p>",
                    "solution": "<p>Solution:<img src=\\"https://image_url\\" alt=\\"Solution\\"></p>"
                    }

                    Ensure to:
                    1. Extract each question, all 4 options, answer, and solution from the given HTML at the start. 
                    2. Format each extracted part into JSON as shown in the example.
                    3. Preserve the <img> tags and any other HTML tags within the content.
                    4. Provide the output strictly in JSON format without any additional text or formatting markers.
                    5. Carefully understand the html at the top and make sure it is extracted in question, answer, options and solution with none of these fields being blank. 

                    """
                user_prompt_json = {
                        
                        "Role": "You are an expert HTML formatter specializing in converting text into well-structured HTML code for educational purposes. Your task is to Understand the given input and fetch questions, options , correct answer and the solution perfectly and  format each question, its options, the correct answer, and the solution in a JSON as provided in the prompt. if the content provided is scrape and not much of releveant to make a question you give all the elements of the JSON in empty ",
                        "objective": prompt_json
                    }
                with open("log.txt", "a") as log_file:
                    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{current_time}: Data Result HTML: \033[1;34m{data_res_html}\033[0m\n")
                    log_file.write(f"{current_time}: User Prompt JSON: \033[1;32m{json.dumps(user_prompt_json)}\033[0m\n")

                
                    # print(data_res_html)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: 2021-09 Current date: "+formatted_date
                        },
                        {
                            "role": "user",
                            "content": json.dumps(user_prompt_json)
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.3
                )
                # Print the response
                if response.choices:
                    # Access the content from the response
                    data_res = response.choices[0].message.content
                    # print(content)
                else:
                    print("No response generated.")
                # with open("log.txt", "a") as log_file:
                #     current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
                #     log_file.write(f"{current_time}: Date Result: \033[1;34m{data_res}\033[0m\n")

                data_res = str(data_res)
                with open("log.txt", "a") as log_file:
                    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{current_time}: Date Result: \033[1;34m{data_res}\033[0m\n")
                try:
                    if "```json" in data_res:
                        data_res = data_res.split("```json")[1]
                        data_res = data_res.split("```")[0]
                        # print(data_res)
                        data_res = json.loads(data_res)
                    # try:
                    #     data_res = json.loads(data_res)
                    # except json.JSONDecodeError as e:
                    #     print(f"JSON decoding error: {e}")
                    #     continue
                    if type(data_res) == str:
                        data_res = json.loads(data_res)
                    
                    
                    if isinstance(data_res, str):
                        try:
                            data_res = json.loads(data_res)
                        except json.JSONDecodeError as e:
                            print(f"JSON decoding error (2nd attempt): {e}")
                            continue
                    
                    api_to_send_questions = "https://p1.edurev.in/Tools/PDF_TO_QuizQuestions_Automation"

                    if  'questions' in data_res :
                        for question in data_res["questions"]:
                            for i, option in enumerate(question["options"]):
                                # Find the option number using regex
                                match = re.search(r"Option ([A-Z0-9]):", option)
                                if match:
                                    option_number = match.group(1)
                                    # Replace the option number with an empty string
                                    question["options"][i] = re.sub(r"Option [A-Z0-9]:", "", option)
                            if isinstance(question, dict):
                                res = {
                                    "quizId" : quizId,
                                    "quizGuid" : quizGuid,
                                    "api_token" : "45b22444-3023-42a0-9eb4-ac94c22b15c2",
                                    "result" : {
                                        "questions":[]
                                    }
                                }
                                res["result"]["questions"].append(question)
                                # print(res)
                                send_question = requests.post(api_to_send_questions, json=res)
                                if send_question.status_code == 200:
                                    print("Question sent successfully!")
                                # send response on api 
                                result["questions"].append(question)
                    else : 
                        for i, option in enumerate(data_res["options"]):
                            # Find the option number using regex
                            match = re.search(r"Option ([A-Z0-9]):", option)
                            if match:
                                option_number = match.group(1)
                                # Replace the option number with an empty string
                                data_res["options"][i] = re.sub(r"Option [A-Z0-9]:", "", option)
                        res = {
                            "quizId" : quizId,
                            "quizGuid" : quizGuid,
                            "api_token" : "45b22444-3023-42a0-9eb4-ac94c22b15c2",
                            "result" :{
                                "questions":[]
                            }
                        }
                        res["result"]["questions"].append(data_res)
                        # print(res)

                        send_question = requests.post(api_to_send_questions, json=res)
                        # print(send_question.json())
                        if send_question.status_code == 200:
                            print("Question sent successfully!")
                        # send response on api
                        result["questions"].append(data_res)
                except Exception as e :
                    print(str(e))
                
                # result["questions"].append(data_res)
                
                # print(result)
                # return result 
            except Exception as e:
                print(e)
        try:
            filepath = filepath[:-5] + "_result.json"
            for res in result["questions"]:
                try:
                    correct_options = ["A", "B", "C", "D"]
                    for i, option in enumerate(res["options"]):
                            # Find the option number using regex
                            match = re.search(r"Option ([A-Z0-9]):", option)
                            if match:
                                option_number = match.group(1)
                                # Replace the option number with an empty string
                                res["options"][i] = re.sub(r"Option [A-Z0-9]:", "", option)
                            if res["options"][i] in res["answer"]:
                                res["answer"] =correct_options[i]
                except Exception as e:
                    print("error while extracting the data from Each Question: "+ str(e))
                res['answer'] = re.sub(r"Answer:", "", res['answer'])
                res['answer'] = re.sub(r"answer:", "", res['answer'])
                res['question'] = re.sub(r"Question:", "", res['question'])
                res['question'] = re.sub(r"question:", "", res['question'])
                res['solution'] = re.sub(r"Solution:", "", res['solution'])
                res['solution'] = re.sub(r"solution:", "", res['solution'])
            with open ("result.json", 'w') as file:
                json.dump(result, file, indent=4)
            return result
        except Exception as e:
            print("error while extracting the data from DATA RESULT JSON: "+ str(e))
    except Exception as e:
        print(e)
        # print("hey")

        return {"data": "Some Error Happend! "}

# def get_result(file_path,role,prompt):
def get_result(file_path,quizId,quizGuid):
    # file_path = r'/var/www/html/images/e7e2b8b9-0d87-41e5-843c-54971856f37e.html'
    # output_path = location_of_images + str(uuid.uuid4()) + ".html"
    
    html_content = read_html(file_path)
    return format_questions(file_path,str(html_content),quizId,quizGuid)
    # return format_questions(file_path,str(html_content),role,prompt)

