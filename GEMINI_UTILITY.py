import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import base64
from PIL import Image
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from io import BytesIO
from langchain.chains import LLMChain




working_directory = os.path.dirname(os.path.abspath(__file__))
config_file_path = f"{working_directory}/config.json"
config_data = json.load(open(config_file_path))
OPENAI_API_KEY = config_data["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- 5. Initialize LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def translate(input_language, output_language, input_text):
    # Construct the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that only translates text from {input_language} to {output_language}."),
            ("human", "{input_text}")
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "input_language": input_language,
            "output_language": output_language,
            "input_text": input_text  # Fixed: Changed from 'input' to 'input_text'
        }
    )
    return response.content


# Initialize the model globally (so it doesn't reload on every call)
LLm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def get_response_from_gpt(user_message, chat_history):
    """
    Appends user message, gets a response from the model, appends it, and returns the response text.
    """
    # Add user message
    chat_history.append(HumanMessage(content=user_message))

    # Get model response
    response = LLm(chat_history)

    # Add assistant response
    chat_history.append(AIMessage(content=response.content))

    return response.content, chat_history



vision_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
def get_image_caption(image: Image.Image) -> str:
    prompt = "Write a short poetic caption about this image. And also tell me what exactly this image is."
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    # LangChain-compatible ChatOpenAI call with image input
    from langchain_core.messages import HumanMessage as LC_HumanMessage

    response = vision_llm([
        LC_HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        )
    ])

    return response.content

def get_text_embedding(text: str):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vector = embedder.embed_query(text)
    return vector