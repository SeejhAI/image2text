# image to text 
# Use a pipeline as a high-level helper
from transformers import pipeline
from dotenv import find_dotenv , load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate

load_dotenv(find_dotenv())

def img2text(url):
    # image_to_text
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)
    print(text)
    return(text)


def generate_story(text):
    image_to_text =  pipeline("text-generation", model="Tincando/fiction_story_generator")
    text = image_to_text(text)
    print(text[0]['generated_text'])
    return(text) 
scenrio = img2text('ishaque.jpg')
scenrio[0]["generated_text"]
generate_story(scenrio[0]["generated_text"])