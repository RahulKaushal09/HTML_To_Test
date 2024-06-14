from bs4 import BeautifulSoup
import re
import uuid
from gpt_test_creation import get_result

def read_html(file_path):
    """Reads HTML content from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content
location_of_images = "/var/www/html/images/"
PUBLIC_IP = "http://52.139.218.113/images/"
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import urllib.parse
import time
import os

import uuid


def get_image(latex_code):
        uuid_image_path = location_of_images + str(uuid.uuid4()) + ".png"
    # Prepare the HTML content with the provided MathJax string
        html_text = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>MathJax Image Automation</title>
            <style>
                body, html {{
                    margin: 10px;
                    padding: 5px;
                    overflow: hidden;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                #math-content {{
                    position: relative;
                    left:20px;
                    width: 100%;
                    padding: 15px;
                    text-align: center;
                    transform-origin: center center;
                }}
            </style>
        </head>
        <body>
            <div id="math-content">
                {latex_code}
            </div>
        </body>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js" integrity="sha256-4

    <script>

                window.onload = function() {{
                    
                    document.getElementById('math-content').style.height =
                        document.querySelector('.MathJax').getBoundingClientRect().height+5 + 'px';
                }};
            </script>
        </html>
        '''
        # print("******************************")
        # print(html_text)
        # print("******************************")
        encoded_html = urllib.parse.quote(html_text)
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        
        user_agent_string = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        options.add_argument(f"user-agent={user_agent_string}")
        options.add_argument('--headless')  # Run in background
        options.add_argument("window-size=1400,1500")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("start-maximized")
        options.add_argument("enable-automation")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-dev-shm-usage")  # Run in background
        # Setup WebDriver
        
        options.headless = True  # Enable headless mode if no GUI is needed
        driver = webdriver.Chrome(service=service, options=options)

        # Use the data URI scheme to load the HTML content directly
        driver.get(f"data:text/html;charset=utf-8,{encoded_html}")

        # Wait for MathJax to render (adjust the timeout as necessary)
        # WebDriverWait(driver, 10).until(lambda x: driver.execute_script("hey"))
            # WebDriverWait(driver, 10).until(lambda x: driver.execute_script("return MathJax.isReady();"))

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "math-content"))
        )

        # Wait a bit for rendering to stabilize
        # time.sleep(2)

        # Find the MathJax container element
        math_element = driver.find_element(By.ID, "math-content")
        # math_element = driver.find_all(By.TAG,"body")

        # math_element = driver.find_element(By.CLASS_NAME, "math-content")
        # math_element = driver.find_element(By.CLASS_NAME, "math-tex")

        # driver.save_screenshot(uuid_image_path)
        # Take a screenshot of just the MathJax element
        math_element.screenshot(uuid_image_path)

        # Cleanup
        driver.quit()
        return True
    
# def latex_to_image(i,soup, latex_expression):
#     # Set up the figure and axis
#     fig, ax = plt.subplots(figsize=(4, 2))  # Adjusted size for better readability
    
#     # Render LaTeX expression
#     ax.text(0.5, 0.5, r"$%s$" % latex_expression, size=20, ha='center', va='center')
#     ax.axis('off')
#     image_url = location_of_images + str(uuid.uuid4()) + ".png"
#     # Save the figure
#     # image_url = f"latex_image{i}.png"
#     fig.savefig(image_url, bbox_inches='tight', pad_inches=0.0)
    
#     # Display the figure
#     plt.show()
#     plt.close(fig)
#     image_url = image_url.replace(location_of_images, PUBLIC_IP)
#     # Create a new img tag with the saved image path
#     # new_img_tag = soup.new_tag("img", src=image_url)
#     # print(new_img_tag)
#     return image_url

# Example usage
soup = BeautifulSoup("<html><body></body></html>", "html.parser")
# latexcode = r"\[+ {NaNO}_{2} + {CH}_{3}COOH \rightarrow X\]"
# latexcode = r"{NaNO}_{2} + {CH}_{3}COOH \rightarrow X"
def get_html_content(element):
        return ''.join(str(child) for child in element.children)
def extract_questions_and_solutions(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    questions = {}
    solutions = {}
    current_solution_key = None
    i=0
    # Parse all paragraphs, handling questions and solutions differently
    
        
    for para in soup.find_all('p'):
        try:
            text = para.get_text(strip=True)
            latex_spans = para.find_all('span', class_='math display')
            
            for span in latex_spans:
                latexcode = span.get_text(strip=True)[2:-2]
                # image_url = latex_to_image(i,soup, latexcode)
                image_url = get_image(latexcode)
                new_img_tag = soup.new_tag("img", src=image_url)
                span.replace_with(new_img_tag)
                i += 1
            # if(text.startswith("\[") and text.endswith("\]")):
            #     latexcode = text[2:-2]
            #     # new_image_tag = latex_to_image(i,soup, latexcode)
            #     # text.getparent().replace_with(str(new_image_tag))
            #     print(para)
            #     # para.replace_with(str(new_image_tag))
            #     # text = str(new_image_tag)
            #     continue
                
            question_match = re.match(r'^(Q\.(\d+)|Question (\d+)|(Q\(\d+)|In which|Q\d+ - \d{4} \(\d{2} [A-Za-z]+ Shift \d\))', text)
            solution_match = re.search(r'(Explanation (\d+):|Solution (\d+):|Q\d+)', text)

            # Identify question and initialize structure
            if question_match:
                q_num = question_match.group(2) or question_match.group(3) or re.match(r'Q\d+', text).group() or 'unknown'
                questions[q_num] = {'question': get_html_content(para), 'options': [], 'solution': ''}

            # Identify start of a new solution
            elif solution_match:
                s_num = solution_match.group(2) or solution_match.group(3) or re.match(r'Q\d+', text).group() or 'unknown'
                current_solution_key = s_num
                solutions[current_solution_key] = get_html_content(para)

            # Accumulate text to the current solution
            elif current_solution_key:
                solutions[current_solution_key] += ' ' + get_html_content(para)
            # Handle options for the latest question
            else:
                if questions:
                    last_question_key = list(questions.keys())[-1]
                    questions[last_question_key]['options'].append(get_html_content(para))
        except Exception as e:
            print(str(e))
    # Map solutions to questions
    for q_num, q_data in questions.items():
        if q_num in solutions:
            q_data['solution'] = solutions[q_num]

    return questions

def rebuild_html(questions):
    new_html = ""
    for q_num, q_data in questions.items():
        new_html += f"<p>{q_data['question']}</p>"
        for option in q_data['options']:
            new_html += f"<p>{option}</p>"
        new_html += f"<p>{q_data['solution']}</p>"
        new_html += "**********"
    
    return new_html

def map_questions(file_path):
    # file_path = r'/var/www/html/images/71713aa9-257e-4412-aca2-92bc1d0e6beb/output.html'
    output_path = location_of_images + str(uuid.uuid4()) + ".html"
    
    html_content = read_html(file_path)
    questions = extract_questions_and_solutions(html_content)
    new_html = rebuild_html(questions)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(new_html)

    print(f"Output saved to {output_path}")
    html_code_path = output_path.replace(location_of_images, PUBLIC_IP)
    print(f"Output saved to {html_code_path}")
    return get_result(output_path)

