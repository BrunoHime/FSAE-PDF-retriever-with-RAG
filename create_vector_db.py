"""
Notes: if there is already a vector database created (folder named qdrant_path), you don't have to run this code, unless you want to update
the database or change eletric vehicle rules to combustion engine rules or vice versa.


Important functions used:

get_toc()

document.get_toc() returns a list of lists, where each list contains the following elements:
[level, title, page]

GR - General Rules --> Level 1
    GR.1 Formula SAE Competition Objective --> Level 2
        GR.1.1 Competition Objective --> Level 3
    GR.2 Organizer Authority --> Level 2
        GR.2.1 Organizer Authority --> Level 3
AD - Administration --> Level 1

Example: [1, 'GR - General Rules', 1] --> Level 1 in page 1



to_markdown()

This function converts a PDF file to a markdown file. It returns a string containing the markdown text.
This allows for the LLM to understand bold, italic, and underline text, as well as headers, images and tables.

The first argument is the path to the PDF file. The second argument is a list of page numbers to be converted to markdown.
The third argument is a boolean value that indicates whether the images should be written to disk.

"""

import fitz  # PyMuPDF
import pymupdf4llm
import os
import shutil


# Path to your PDF file
pdf_path = "FSAE_Rules_2024.pdf"
document = fitz.open(pdf_path)

# document_subtopics is a dictionary where the key is the title of the subtopic and the value is the page number
# We will create the chunks of text based on the subtopics (level 3)
# The document_topics and document_sections lists will be used to remove the topics (level 2) and sections (level 1)
    # in case they appear inside the text of document_subtopics
document_subtopics = {}
document_topics = []
document_sections = []

# Cleaning the text extracted from the pdf
# In some cases, when extracting with the .get_toc() the text comes with double spacing, which can affect the text extraction
# Corrects double spacing with the split method without affecting future PDFs with possible correct spacing.
# Also corrects the spacing of the comma, which is sometimes followed by a space and sometimes not.
for item in document.get_toc():
    if item[0] == 3:
        correct_item = " ".join(item[1].split("  ")).replace(" ,", ",").strip()
        document_subtopics[correct_item] = item[2]
    if item[0] == 2:
        correct_item = " ".join(item[1].split("  ")).replace(" ,", ",").strip()
        document_topics.append(correct_item)
    if item[0] == 1:
        correct_item = " ".join(item[1].split("  ")).replace(" ,", ",").strip()
        document_sections.append(correct_item)


def remove_md_topics_and_sections(md_text: str) -> str:

    """Removing the topics (level 2) and sections (level 1) from the text

    Args:
        md_text (str): Markdown text to be cleaned

    Returns:
        str: Markdown text without the topics and sections
    """    

    for item in document_topics:
        md_item = item.split(" ", 1)
        md_item = " ".join(["### " + c.upper() for c in md_item])
        if md_item in md_text:
            md_text = md_text.replace(md_item, "")
    for item in document_sections:
        md_item = "## " + item.upper()
        if md_item in md_text:
            md_text = md_text.replace(md_item, "")

    return md_text



list_of_subtopics = list(document_subtopics.keys())

texts_to_vectorize = {}

for index, subtopic in enumerate(list_of_subtopics):

    # Get the markdown text from the page of the current subtopic. Note that the page number is 0-based
    current_page = document_subtopics[subtopic]
    md_text = pymupdf4llm.to_markdown(pdf_path, pages=[current_page - 1], write_images=True)

    # Removing possible new topics and sections from the middle page
    md_text = remove_md_topics_and_sections(md_text)
    
    # Changing the current subtopic text to the markdown format, and finding its position
    md_subtopic = subtopic.split(" ", 1)
    md_subtopic = " ".join(["**" + c + "**" for c in md_subtopic])
    current_subtopic_pos = md_text.find(md_subtopic)

    # Check if is not the last current subtopic
    if index != len(list_of_subtopics) - 1:
        
        # Changing the next subtopic text to the markdown format, and finding its position	
        next_subtopic = list_of_subtopics[index + 1]
        md_next_subtopic = next_subtopic.split(" ", 1)
        md_next_subtopic = " ".join(["**" + c + "**" for c in md_next_subtopic])
        next_subtopic_pos = md_text.find(md_next_subtopic)

        # Check if the next subtopic is in the same page as the current subtopic
        if current_page == document_subtopics[list_of_subtopics[index + 1]]:
            # Extracting the text between the current and next subtopic
            text = md_text[current_subtopic_pos:next_subtopic_pos]
            texts_to_vectorize[text] = {"text": text, "page": current_page}

        # The text of the current subtopic might be split between two or more pages
        else:
            # Removing unnecessary information from the bottom of the page
            last_pos = md_text.find("Formula SAE® Rules")

            # Extracting the text from the current subtopic to the end of the page
            text = md_text[current_subtopic_pos:last_pos]

            # Check if the next page has a continuation of the current subtopic
            # Also, the subtopic might be split in more than one page distance, this loop will add the text until the next subtopic is found
            # Subtracting 1 from the current page because the pages are 0-based and each loop will add 1 to the next_page
            next_page = current_page - 1
            while True:
                next_page += 1

                # Check if it is the last page of the document
                # Current page must be equal to the page_count because the pages are 0-based
                if next_page == document.page_count:
                    break

                md_next_pg_text = pymupdf4llm.to_markdown(pdf_path, pages=[next_page], write_images=True)
                
                # if the index of next_subtopic_pos is -1, it means that the next subtopic is not in the current page
                next_subtopic_pos = md_next_pg_text.find(md_next_subtopic)

                if next_subtopic_pos == -1:
                    last_pos = md_next_pg_text.find("Formula SAE® Rules")
                    text += md_next_pg_text[:last_pos]
                else:
                    break

            # Removing possible new topics and sections from the top page and resetting the next_subtopic_pos
            md_next_pg_text = remove_md_topics_and_sections(md_next_pg_text)
            next_subtopic_pos = md_next_pg_text.find(md_next_subtopic)

            text += md_next_pg_text[:next_subtopic_pos]
            texts_to_vectorize[text] = {"text": text, "page": current_page}

    # Last subtopic
    else:
        # Finding the position of where the last subtopic's text ends
        last_pos = md_text.find("Formula SAE® Rules")
        # Extracting the text from the current subtopic to the end of the page
        text = md_text[current_subtopic_pos:last_pos]
        texts_to_vectorize[text] = {"text": text, "page": current_page}



# Since this pdf is about the FSAE rules for all teams, you can filter the combustion engine rules and electric vehicle rules
# depending on the category of your team

your_team_category = "Electric Vehicle"  # or "Combustion Engine"

if your_team_category == "Electric Vehicle":
    # For eletric vehicle rules
    texts_to_vectorize = {k: v for k, v in texts_to_vectorize.items() if not v["text"][:4] == "**IC"}
elif your_team_category == "Combustion Engine":
    # For combustion engine rules
    texts_to_vectorize = {k: v for k, v in texts_to_vectorize.items() if not v["text"][:4] == "**EV"}


# Move all images to a new folder called "images"

# Get all files in the current directory
files_in_directory = os.listdir()
filtered_files = [file for file in files_in_directory if file.endswith(".png")]

# Move all image files to the new directory
for file in filtered_files:
    shutil.move(file, 'static/images')



"""
Creating the RAG model

We will create a vector database using the Qdrant library to store the embeddings of each subtopics we extracted from the PDF.
The embeddings will be generated using the Google's generative AI embeddings.

The user will input a question, and the model will return the top 10 most similar subtopics to the question, that will be 
used as a context to the LLM model.


"""

# Libraries for the embeddings and the Qdrant database
import qdrant_client
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Setting the Google API key
GOOGLE_API_KEY = 'your-api-key'
genai.configure(api_key=GOOGLE_API_KEY)

# Connect with Google's generative AI embeddings using LangChain
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)


# Transform the subtopics into lists and implement the embeddings
payload = list(texts_to_vectorize.values())

subtopics = list(texts_to_vectorize.keys())
subtopics = embeddings.embed_documents(subtopics)

index = list(range(len(subtopics)))

# Create a Qdrant client pointing to the path where the vector database will be stored locally.
client = qdrant_client.QdrantClient(path="qdrant_path")

# If it does not exist yet, create the vector database using Qdrant
if not client.collection_exists("fsae_rules"):
    # Create a collection called "fsae_rules" with 768 columns (size of the embedding vector)
    # Use cosine distance to calculate the similarity between vectors
    client.create_collection(
        collection_name="fsae_rules",
        vectors_config=models.VectorParams(size=768,
                                            distance=models.Distance.COSINE
                                                )
    )

    # Add to the vector database the subtopics with the embedding, the payloads with the pages and subtopics with markdown, and their respective index
    client.upsert(
        collection_name="fsae_rules",
        points=models.Batch(
            ids=index, 
            vectors=subtopics,
            payloads=payload
        )
    )



