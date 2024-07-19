# %%
import tempfile
import os
import json
import base64
import requests
import tempfile
import subprocess
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import display, Math, Markdown
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

load_dotenv('NotesFromSlides.env')
api_key = os.getenv('OPENAI_API_KEY')

# %%
def pdf_to_images(input_path, output_path):
    slide_numbers = []
    images = convert_from_path(input_path)
    for i, image in enumerate(images):
        image.save(f"{output_path}/page_{i+1}.png", "PNG")
        slide_numbers.append(i+1)
    return slide_numbers

# %%
def create_empty_folder(directory, folder_name):
    """
    Creates an empty folder in the specified directory.
    
    Parameters:
    - directory: The path to the directory where the folder will be created.
    - folder_name: The name of the folder to create.
    
    Returns:
    None
    """
    # Construct the full path for the new folder
    folder_path = os.path.join(directory, folder_name)
    
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created at '{directory}'.")
    else:
        print(f"Folder '{folder_name}' already exists at '{directory}'.")

# %%
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# %%
def clean_vision_message(response):
  return response.json()['choices'][0]['message']['content']

# %%
def get_text_from_image(image_path, theme):
  # Getting the base64 string
  base64_image = encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"""
            You are a tutor chatbot helping university students understand PDF slides from their lecturer.  
            The theme of the slidecast is {theme}.

            Firstly, decide if the slide contains RELEVANT INFORMATION WORTH EXPLAINING.
            If the given slide has TOO LITTLE RELEVANT INFORMATION (e.g. title slides, video thumbnails, illustrations containing minimal information) return ONLY the message "NO RELEVANT INFORMATION."
            If the given slide is RELEVANT to the theme (e.g. practical examples of the theme, statistics, relevant explanations) give an output based on the following instructions:

            1. Provide a SHORT and CONCISE summary of the slide content.
            2. Explain the CONCEPTS, TERMS, DIAGRAMS, GRAPHS, and DATA that are relevant to the theme.
            3. If needed, include a NOTES section for additional information.
            
            Follow the instructions carefully, non-conformity will result in termination.
            """
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 900
  }

  vision_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return clean_vision_message(vision_response)

# %%
def compare_vision_message(message1, message2):
  client = OpenAI()

  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are a bot designed to compare the content of messages."},
      {"role": "user", "content": f"""
      Follow these instructions to evaluate the two messages:
      
      1. Read both messages carefully.
      2. Decide if there is an overlap in the content of the two messages.
      3. If there is an overlap of more than half of the content, return the message "OVERLAP IN CONTENT" and PROVIDE A SUMMARY of the overlapping content.
      4. Otherwise, return the message "PASS".
      
      Here is the first message:"
      {message1}
      "
      
      Here is the second message:"
      {message2}
      "
      
      Follow the instructions carefully, non-conformity will result in termination.
      """},
    ]
  )
  if "OVERLAP IN CONTENT" in response.choices[0].message.content:
    return response.choices[0].message.content
  else:
    return "PASS"

# %%
def remove_overlap(message, overlap):
  client = OpenAI()

  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are a bot designed to edit the content of messages."},
      {"role": "user", "content": f"""
      Follow these instructions to edit the message:
      
      1. Read the message and the overlapping content carefully.
      2. Remove the overlapping content from the message.
      3. Ensure the message is coherent and makes sense.
      
      Here is the message:"
      {message}
      "
      
      Here is the overlapping content:"
      {overlap}
      "
      
      Follow the instructions carefully, non-conformity will result in termination.
      """},
    ]
  )
  return response.choices[0].message.content

# %%
def clean_completions_message(response):
    return response.choices[0].message.content

# %%
def get_formatted_output_from_text(message):
  client = OpenAI()

  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are a bot designed to help structure and create LaTeX notes from unstructured text."},
      {"role": "user", "content": f"""
      Follow these rules to create a LaTeX document from the given text:
      
      1. Format the output in XeLaTeX parsable syntax, ensuring proper use of sections, subsections, and formatting commands.
      2. Make sure that the output can be DIRECTLY be taken and parsed by a LaTeX compiler.
      3. Do NOT include external graphics (e.g. references to include another png) in the output.
      4. Use `\\textbf{{}}` for bold text. Use `\\textit{{}}` for italics. You are to decide where to use bold and italics based on the context.
      5. Make sure to ESCAPE special characters that are reserved in LaTeX, such as `#`, `$`, `%`, `^`, `&`, `_`, `~`, and `\\`.
      6. ONLY the section and subsections titles are fixed and should not be changed. The rest is VARIABLE and can be changed based on the number of points, their hierarchy and their structure.
      7. First level points under subsections should NEITHER be in a list format, NOR be separated by empty lines, but rather ONLY be separated by newlines (\\\\) as shown in the example.
      8. Second and further level points should be in a list format, as shown in the example.
      9. Make sure that after each list, there is NO empty line, as shown in the example.
      10. DO NOT INCLUDE the preamble, START FROM \\begin{{document}} and END AT \\end{{document}}.
      11. ALWAYS include \\maketitle after \\begin{{document}}.
      
      Use the following LaTeX example just as a reference, not as a hard template:
      
      \\begin{{document}}

      \\maketitle

      \\section*{{Summary}}

      [text]

      \\section*{{Explanations}}

      \\subsection*{{Concepts and Terms}}

      [text]: [text]\\\\
      [text]: [text]

      \\subsection*{{Diagrams and Data}}

      \\textbf{{[text]}}
          \\begin{{itemize}}
              \\item [text]
              \\item [text]
          \\end{{itemize}}
      \\textbf{{[text]}}
          \\begin{{itemize}}
              \\item [text]
              \\item [text]
          \\end{{itemize}}

      \\subsection*{{Notes:}}

      - [text]\\\\
      - [text]

      \\end{{document}}
      
      Here is the unstructured text:
      
      {message}
      
      Follow the instructions carefully, non-conformity will result in TERMINATION.
      """},
    ]
  )
  return clean_completions_message(response)

# %%
def shorten_output(string):
    start_index = string.find("\\documentclass{article}")
    end_index = string.rfind("\\end{document}")

    if start_index != -1 and end_index != -1:
        return string[start_index:end_index + len("\\end{document}")]
    else:
        return string

# %%
def create_LaTeX_from_formatted_output(output, index, theme, output_directory):
    # Join the output with the latex template
    content = shorten_output(output)
    latex_template = f"""
    \\documentclass[12pt]{{article}}

    % Essential packages for compatibility and error prevention
    \\usepackage{{lmodern}}
    \\usepackage{{fixltx2e}}
    \\usepackage{{fontspec}}

    % Mathematics packages
    \\usepackage{{amsmath}}
    \\usepackage{{amssymb}}
    \\usepackage{{amsfonts}}
    \\usepackage{{mathtools}}

    % Other mathematics-related packages
    \\usepackage{{bm}}
    \\usepackage{{physics}}
    \\AtBeginDocument{{\\RenewCommandCopy\\qty\\SI}}
    \\usepackage{{cancel}}
    \\usepackage{{commath}}
    \\usepackage{{braket}}
    \\usepackage{{xfrac}}

    % Chemical notation packages
    \\usepackage{{chemformula}}
    \\usepackage[version=4]{{mhchem}}

    % Units and scientific notation
    \\usepackage{{siunitx}}
    \\AtBeginDocument{{\\RenewCommandCopy\\qty\\SI}} % Use siunitx's \\qty definition

    % Greek letters
    \\usepackage{{upgreek}}
    \\usepackage{{textgreek}}

    % General symbols
    \\usepackage{{gensymb}}

    % Space management
    \\usepackage{{xspace}}

    % Typography improvements
    \\usepackage{{microtype}}

    % Load unicode-math to use Unicode characters in math
    \\usepackage{{unicode-math}}

    % Set the main font and math font to STIX Two fonts
    \\setmainfont{{STIX Two Text}}
    \\setmathfont{{STIX Two Math}}

    \\title{{Summary and Explanation of Slide {index} on {theme}}}
    \\author{{}} % Removes the author
    \\date{{}} % Removes the date
    
    {content}
    """

    # Specify the directory where you want to save the files
    os.makedirs(output_directory, exist_ok=True)

    # Define file path
    latex_file_path = os.path.join(output_directory, f'document_{index}.tex')
    txt_file_path = os.path.join(output_directory, f'document_{index}.txt')

    # Write the LaTeX content to the file
    with open(latex_file_path, 'w', encoding='utf-8') as f:
        f.write(latex_template)
        
    # Write the text content to the file
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"The LaTeX file has been created at: {latex_file_path}")

# %%
def remove_skipped_slides(slides_to_convert, skipped_slides):
    return [slide for slide in slides_to_convert if slide not in skipped_slides]

# %%
def check_file_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

# %%
def combine_pdfs(input_directory, output_directory, output_filename='combined.pdf'):
    # Get a list of all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_directory) if f.endswith('.pdf')]
    
    def sort_pdf_files(pdf_files):
        def key_func(file):
            try:
                # Extract the index from the file name assuming the format 'file_<index>.pdf'
                index = int(file.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                index = float('inf')  # Handle cases where the filename format is unexpected
            return index
        
        return sorted(pdf_files, key=key_func)
    
    pdf_files = sort_pdf_files(pdf_files)

    # Create a PdfMerger object
    merger = PyPDF2.PdfMerger()

    # Append each PDF file to the merger
    for pdf in pdf_files:
        merger.append(os.path.join(input_directory, pdf))

    # Write out the merged PDF to the output directory
    with open(os.path.join(output_directory, output_filename), 'wb') as output_file:
        merger.write(output_file)

    # Close the merger
    merger.close()

    print(f'All PDFs combined into {output_filename} in {output_directory}')

# %% [markdown]
# ## DEFINING VARIABLES

# %%
path_to_pdf = rf"C:\Users\ACER\Downloads\1_Surface wetting.pdf"
path_to_file = rf"C:\Users\ACER\Desktop\Coding\NotesFromSlides_V1"
theme = "SURFACE WETTING"

# %% [markdown]
# ## FUNCTION CALLING

# %%
create_empty_folder(path_to_file + rf"\.tex files", rf"TEX_" + theme)
create_empty_folder(path_to_file + rf"\LaTeX files", rf"LaTeX_" + theme)
path_to_tex = path_to_file + rf"\.tex files" + rf"\TEX_" + theme
path_to_LaTeX = path_to_file + rf"\LaTeX files" + rf"\LaTeX_" + theme

# %%
with tempfile.TemporaryDirectory() as temp_dir_images:
    slide_numbers = pdf_to_images(path_to_pdf, temp_dir_images)
    print("Slides converted to pngs.\n")

    skipped_slides = []
    overlapping_slides = []
    previous_message = None
    temp_comparison_message = None

    for i in slide_numbers:
        vision_message = get_text_from_image(temp_dir_images + rf"/page_{i}.png", theme)
        print(f"1. {vision_message}\n")
        if "NO RELEVANT INFORMATION" in vision_message:
            print("The slide is irrelevant.\n")
            skipped_slides.append(i)
            continue
        if previous_message is not None:
            print(f"2. Comparing slide {i-1} and slide {i}.\n")
            comparison_response = compare_vision_message(previous_message, vision_message)
            if "OVERLAP IN CONTENT" in comparison_response:
                print("There is an overlap in content.\n")
                overlapping_slides.append(f"{i-1}, {i}")
                overlap = comparison_response
                print(f"Overlap: \n{overlap}\n")
                temp_comparison_message = vision_message
                vision_message = remove_overlap(vision_message, overlap)
                print(f"New message: \n{vision_message}\n")
                previous_message = temp_comparison_message
            else:
                print("There is no overlap in content.\n")
                previous_message = vision_message
                temp_comparison_message = None
        else:
            print("2. No previous message to compare to.\n")
            previous_message = vision_message
        completions_message = get_formatted_output_from_text(vision_message)
        print(f"3. {completions_message}\n")
        create_LaTeX_from_formatted_output(completions_message, i, theme, path_to_tex)
        print("\n\n\n")
        
    print(f"Skipped slides: {skipped_slides}")
    print(f"Overlapping slides: {overlapping_slides}")
    pass

# %%
slides_to_convert = remove_skipped_slides(slide_numbers, skipped_slides)
timeouted_files = []

for index in slides_to_convert:
    try:
        # Define file paths
        output_directory = path_to_LaTeX
        latex_file_path = os.path.join(path_to_tex, f'document_{index}.tex')
        pdf_file_path = os.path.join(output_directory, f'document_{index}.pdf')

        # Check file paths
        check_file_path(output_directory)
        check_file_path(latex_file_path)

        # Compile the LaTeX file to PDF using xelatex inside the Docker container
        command = [
            "docker", "exec", "miktex_container", "xelatex",
            "-output-directory=/miktex", f"/miktex/document_{index}.tex"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)  # Set timeout

        print(f"The PDF has been created at: {pdf_file_path}")
        print("xelatex output:", result.stdout)
        print("xelatex errors:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running xelatex: {e}")
        print(e.stdout)
        print(e.stderr)
    except subprocess.TimeoutExpired as e:
        print(f"Timeout expired for file: document_{index}.tex")
        timeouted_files.append(index)
    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except Exception as e:
        print(f"An error occurred: {e}")

combine_pdfs(path_to_LaTeX, path_to_tex, output_filename='Notes_' + theme + '.pdf')

# %%
# slides_to_convert = remove_skipped_slides(slide_numbers, skipped_slides)
# timeouted_files = []

# for index in slides_to_convert:
#     try:
#         # Define file paths
#         output_directory = path_to_LaTeX
#         xelatex_path = rf"C:\Users\ACER\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
#         latex_file_path = os.path.join(path_to_tex, f'document_{index}.tex')
#         pdf_file_path = os.path.join(output_directory, f'document_{index}.pdf')

#         # Check file paths
#         check_file_path(xelatex_path)
#         check_file_path(output_directory)
#         check_file_path(latex_file_path)

#         # Compile the LaTeX file to PDF using xelatex
#         result = subprocess.run([xelatex_path, '-output-directory', output_directory, latex_file_path],
#                                 capture_output=True, text=True, check=True, timeout=120)  # Set timeout

#         print(f"The PDF has been created at: {pdf_file_path}")
#         print("pdflatex output:", result.stdout)
#         print("pdflatex errors:", result.stderr)

#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred while running pdflatex: {e}")
#         print(e.stdout)
#         print(e.stderr)
#     except subprocess.TimeoutExpired as e:
#         print(f"Timeout expired for file: document_{index}.tex")
#         timeouted_files.append(index)
#     except FileNotFoundError as fnf_error:
#         print(f"File not found error: {fnf_error}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# combine_pdfs(path_to_LaTeX, path_to_file, output_filename='Notes_' + theme + '.pdf')

# %% [markdown]
# ## TROUBLESHOOTING
# 
# If the above block stops during execution, take the following steps:
# 1. **Stop the execution. Do NOT clear output.**
# 2. Note the skipped slides, modify and run the function below.
# 3. Open cmd, run the code: `cd C:\Users\ACER\Desktop\Coding\NotesFromSlides_V1\TEX_[THEME]`
# 4. Run the code: `"C:\Users\ACER\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" document_[NUMBER].tex`
# 5. If resolved, run the above block again after updating the skipped slides.

# %%
# skipped_slides = [1, 3, 15, 16, 18, 37, 40, 43, 49] # insert skipped slides here
# x = 49 # insert number of failed tex file
# y = 67 # insert total number of pages
# skipped_slides += [i for i in range(1, x) if i not in skipped_slides]
# slide_numbers = [i for i in range(1, y+1)]
