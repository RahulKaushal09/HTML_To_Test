import json
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
from gpt_test_creation import get_result
import uuid

from PIL import Image
import json


def get_image(mathjax, uuid_image_path):
    # Prepare the HTML content with the provided MathJax string
    html_text = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <link href="https://fonts.googleapis.com/css?family=Noto+Sans&display=swap" rel="stylesheet">

    <meta charset="UTF-8">
        <title>MathJax Image Automation</title>
        <style>
            body, html {{
                margin: 5px;
                padding: 5px;
                overflow: hidden;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: 'Noto Sans', sans-serif !important;
            }}
            #math-content {{
                
                padding: 2px;
                text-align: center;
                transform-origin: center center;
            }}
        </style>
    </head>
    <body>
        <div id="math-content">
            {mathjax}
        </div>
    </body>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script type="text/javascript"
src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script>

            window.onload = function() {{
                
                document.getElementById('math-content').style.height =
                    document.querySelector('.MathJax').getBoundingClientRect().height+15 + 'px';
            }};
        </script>
    </html>
    '''
    # print("******************************")
    # print(html_text)
    # print("******************************")
    encoded_html = urllib.parse.quote(html_text)
    # service = Service(ChromeDriverManager().install())
    service = Service('/root/.wdm/drivers/chromedriver/linux64/114.0.5735.90/chromedriver-linux64/chromedriver')
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

    # Take a screenshot of just the MathJax element
    math_element.screenshot(uuid_image_path)

    # # Remove background from the image
    # image = Image.open(uuid_image_path)
    # image = image.convert("RGBA")
    # data = image.getdata()

    # new_data = []
    # for item in data:
    #     # Set pixels with white background to transparent
    #     if item[:3] == (255, 255, 255):
    #         new_data.append((255, 255, 255, 0))
    #     else:
    #         new_data.append(item)

    # image.putdata(new_data)

    # # Save the image with transparent background
    # image.save(uuid_image_path, "PNG")

    # Cleanup
    driver.quit()
    return True
def get_table_image(table_container, uuid_image_path):
    # Prepare the HTML content with the provided MathJax string
    html_text = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
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
                
                padding: 10px;
                text-align: center;
                transform-origin: center center;
            }}
        </style>
    </head>
    <body>
        <div id="math-content">
            {table_container}
        </div>
    </body>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js" integrity="sha256-4 
        </script>
        
    </html>
    '''
    # print("******************************")
    # print(html_text)
    # print("******************************")
    encoded_html = urllib.parse.quote(html_text)
    # service = Service(ChromeDriverManager().install())
    service = Service('/root/.wdm/drivers/chromedriver/linux64/114.0.5735.90/chromedriver-linux64/chromedriver')
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
    options.add_argument("--disable-web-security")
    options.add_argument('--allow-file-access-from-files')
    options.add_argument('--font-render-hinting=none')
    options.add_argument("--disable-dev-shm-usage")  # Run in background
    # options.add_argument(f"--font-family='Kruti Dev 010'")  # Set custom font family

    # Setup WebDriver
    
    options.headless = False  # Enable headless mode if no GUI is needed
    driver = webdriver.Chrome(service=service, options=options)

    # Use the data URI scheme to load the HTML content directly
    driver.get(f"data:text/html;charset=UTF-8,{encoded_html}")

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "math-content"))
    )
    math_element = driver.find_element(By.ID, "math-content")
    math_element.screenshot(uuid_image_path)

    # Cleanup
    driver.quit()
    return True



image_folder_path = "/var/www/html/images"
Public_IP = "https://fc.edurev.in/images"

def find_top_most_parent_span(element):
    current = element
    while current.parent and current.parent.name == 'span':
        current = current.parent
    return current
processed_parents = set()
top_most_parents = []
def get_parent(soup):
    top_most_parents =[]
    for element in soup.find_all(lambda tag: tag.name == 'span' and tag.has_attr('data-mathml')):
        top_most_parent = find_top_most_parent_span(element)
        
        # Using id or object identity to avoid duplicates
        parent_id = id(top_most_parent)  # Fallback to object ID if no 'id' attribute
        
        if parent_id not in processed_parents:
            processed_parents.add(parent_id)
            top_most_parents.append(top_most_parent)
            # Mark the parent as 'done'
        
    return top_most_parents
def replace_mathjax_with_images(html_content):
    
    """
    Parse HTML content, identify MathJax/LaTeX, convert to images, and replace original content with images.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for element in soup.find_all('script'):
        element.decompose() 
    for element in soup.find_all('mjx-assistive-mml'):
        element.decompose() 
    for element in soup.find_all('mathml'):
        element.decompose() 
    for element in soup.find_all('asciimath'):
        element.decompose() 
    for element in soup.find_all('latex'):
        element.decompose() 
    for element in soup.find_all('mathmlword'):
        element.decompose() 
    for element in soup.find_all('tsv'):
        element.decompose() 
    for element in soup.find_all('table-markdown'):
        element.decompose() 
   
    elements = soup.find_all('nobr')
    for element in elements:
        element.extract()
    attribute = ""
    # Adjust this selector based on your HTML structure
    data_ = {"content":[]}
    math_elements = soup.find_all("mjx-container")
    if (len(math_elements) == 0) :
        math_elements = soup.find_all('span',class_='math-tex')
        # print("***********************************")
        # print(str(soup))
        # print("***********************************")
        if (len(math_elements) == 0) :
            # print("check1")
            
            math_elements = get_parent(soup)
            
            if (len(math_elements) == 0) :
                return str(soup)
            else:
                attribute = "data-mathml"

        else:
            attribute = "math-tex"
            # print(math_elements)        
    else:
        attribute = "mjx-container"
    for elements in math_elements:
        uuid_str = str(uuid.uuid4())
        # Assuming the LaTeX string can be directly obtained from the element's text
        image_url = f"{image_folder_path}/{uuid_str}.png"
        # element = elements.get('data-mathml')
        if(attribute == "mjx-container" ):
            element = elements
        if get_image(element, image_url):
            # print(image_url)
            image_url =f"{Public_IP}/{uuid_str}.png"
            # print(soup)
            new_img_tag = soup.new_tag("img", src=image_url)
            # Create a dictionary to store the element and image URL
            if os.path.exists('data.json'):
                with open('elements_to_img.json', 'r') as file:
                    data_ = json.load(file)
                    print(data_['content'])
            
            replacement = {
                "element": str(elements),
                "image_url": image_url
            }
            # Append the replacement dictionary to the list of replacements
            data_["content"].append(replacement)
            
            with open('elements_to_img.json', 'w') as file:
                json.dump(data_, file)
            elements.replace_with(new_img_tag)
        
    # print(soup.prettify())
    return str(soup)

def replace_tables_with_images(html_content):
    """
    Parse HTML content, identify <table> elements, convert to images, and replace original <table> elements with images.

    Parameters:
    - html_content: A string containing the HTML content.
    - image_folder_path: The file system path where images should be stored.
    - Public_IP: The public IP address or domain where the images will be accessible.

    Returns:
    A string of the modified HTML content with <table> elements replaced by <img> elements linking to images of the tables.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table_elements = soup.find_all('table')
    
    for table in table_elements:
        uuid_str = str(uuid.uuid4())
        image_url = f"{image_folder_path}/{uuid_str}.png"

        # You would need to implement the get_table_image function. It should capture the table element,
        # convert it to an image, and save it to the provided image_url.
        if get_table_image(table, image_url):
            public_image_url = f"{Public_IP}/{uuid_str}.png"
            print(public_image_url)
            new_img_tag = soup.new_tag("img", src=public_image_url)
            table.replace_with(new_img_tag)

    return str(soup)


# Function to map MathML to LaTeX

# Convert the example HTML to LaTeX

# Example HTML content
def excelRun(html_content):
    
    # html_content = """<h3 ng-bind="vernacularStaticTextContent['ts_ques_detail'][8]" class="ng-binding">Explanation:</h3> <p> <span class="question_exp ng-binding" ng-bind-html="toTrustedHTML(show_single_data[6])"><p>Given-</p><p><span class="mathMlContainer" contenteditable="false"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="2" style="font-size: 119.5%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mfrac><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-mfrac space="3"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mi>A</mi><mi>B</mi></mfrac><mo>+</mo><mfrac><mi>B</mi><mi>A</mi></mfrac><mo>=</mo><mn>2</mn></math></mjx-assistive-mml></mjx-container></span><br></p><p><span class="mathMlContainer" contenteditable="false"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="3" style="font-size: 119.5%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c21D2"></mjx-c></mjx-mo><mjx-mfrac space="4"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-mfrac space="3"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c31"></mjx-c></mjx-mn></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mstyle size="s"><mjx-mfrac><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac></mjx-mstyle></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c31"></mjx-c></mjx-mn><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-mfrac space="3"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c31"></mjx-c></mjx-mn></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c31"></mjx-c></mjx-mn></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>⇒</mo><mfrac><mi>A</mi><mi>B</mi></mfrac><mo>+</mo><mfrac><mn>1</mn><mstyle display=""><mfrac><mi>A</mi><mi>B</mi></mfrac></mstyle></mfrac><mo>=</mo><mn>1</mn><mo>+</mo><mfrac><mn>1</mn><mn>1</mn></mfrac></math></mjx-assistive-mml></mjx-container></span></p><p><span class="mathMlContainer" contenteditable="false"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="4" style="font-size: 119.5%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c21D2"></mjx-c></mjx-mo><mjx-mfrac space="4"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c31"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>⇒</mo><mfrac><mi>A</mi><mi>B</mi></mfrac><mo>=</mo><mn>1</mn></math></mjx-assistive-mml></mjx-container></span><br></p><p><span class="mathMlContainer" contenteditable="false"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="5" style="font-size: 119.5%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c21D2"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>⇒</mo><mi>A</mi><mo>=</mo><mi>B</mi></math></mjx-assistive-mml></mjx-container></span><br></p><p><span class="mathMlContainer" contenteditable="false"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="6" style="font-size: 119.5%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c21D2"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2212"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="3"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c30"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>⇒</mo><mi>A</mi><mo>-</mo><mi>B</mi><mo>=</mo><mn>0</mn></math></mjx-assistive-mml></mjx-container></span><br></p></span></p>"""
    html_res = replace_mathjax_with_images(html_content)
    html_res = replace_tables_with_images(html_res)
    # print(html_res)
    return html_res
def extract_qa(content):
    result=""
    contents = content.replace("<ol","**********\n<ol")
    first = True
    for content in contents:
        if first:
            result+=content
        else:
            result+="**********"+content
    # print(result)
    return result
# file_path_ = "/root/pdfToTest/JEE_MATHPIX_pro.html"
# def preprocess_html_code(file_path,role,prompt):
def preprocess_html_code(file_path,quizId,quizGuid):
    """Reads HTML content from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    if 'Edurev 1' in html_content:
            html_content = html_content.replace("Edurev 1","Edurev1")
    result_html_content = excelRun(html_content)
    # result_html_content = extract_qa(result_html_content)
    with open(file_path, "w") as file:
        file.write(result_html_content)
    print(file_path)
    # return html_content
    return get_result(file_path,quizId,quizGuid)
    # return get_result(file_path,role,prompt)
# read_html(file_path)
# with open("/root/pdfToTest/JEE_result.html", 'r', encoding='utf-8') as file:
#         result_html_content = file.read()
# result_html_content = extract_qa(result_html_content)
# with open("/root/pdfToTest/JEE_result_processed.html", "w") as file:
#     file.write(result_html_content)