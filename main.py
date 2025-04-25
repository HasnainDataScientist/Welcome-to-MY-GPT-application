import os
import streamlit as st
from langchain_openai import ChatOpenAI
from streamlit_option_menu import option_menu
from GEMINI_UTILITY import translate
from langchain.schema import HumanMessage, SystemMessage
from GEMINI_UTILITY import get_response_from_gpt
from PIL import Image
from GEMINI_UTILITY import get_image_caption
from GEMINI_UTILITY import get_text_embedding

working_directory = os.path.dirname(os.path.abspath(__file__))


# --- 2. Configure Streamlit page ---
st.set_page_config(
    page_title="Welcome to MY-GPT",
    page_icon="🌐",
    layout="centered"
)

with st.sidebar:
    selected = option_menu(
        menu_title="MYGPT AI",  # Title of the sidebar menu
        options=["AI Translator", "Chatbot", "Image Captioning", "Embed Text"],  # Menu items
        icons=["translate", "robot", "image-fill", "file-earmark-text"],  # Icons for the menu items
        menu_icon="GPT",  # Icon for the title
        default_index=0  # Default selected item
    )


if selected == "AI Translator":
    st.title("AI Translator MY-GPT")

    # --- 3. Language Selection ---
    col1, col2 = st.columns(2)

    with col1:
        input_language_list = ["English", "French", "German", "Urdu", "Arabic", "Dutch", "Hindi"]
        Input_language = st.selectbox(label="Input Language", options=input_language_list)

    with col2:
        output_language_list = [x for x in input_language_list if x != Input_language]
        Output_language = st.selectbox(label="Output Language", options=output_language_list)

    # --- 4. Input Text ---
    Input_text = st.text_area("Enter text to translate:")

    # --- 5. Initialize LLM ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # --- 7. Translate Button ---
    if st.button("Translate"):
        if not Input_text.strip():
            st.warning("Please enter text to translate.")
        else:
            translated_text = translate(Input_language, Output_language, Input_text)
            st.success(translated_text)


if selected == "Chatbot":
    st.title("💬 MY-GPT Chatbot")

    # --- Initialize chat history ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful, conversational AI assistant.")
        ]

    # --- Reset Chat Button ---
    if st.button("🔁 Reset Chat"):
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful, conversational AI assistant.")
        ]
        st.experimental_rerun()

    # --- Display Chat History ---
    for msg in st.session_state.chat_history[1:]:  # Skip system message
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
            st.markdown(f"{avatar} {msg.content}")

    # --- Input and Response ---
    user_message = st.chat_input("Type your message...")

    if user_message:
        with st.chat_message("user"):
            st.markdown(f"🧑‍💻 {user_message}")

        response, updated_history = get_response_from_gpt(user_message, st.session_state.chat_history)
        st.session_state.chat_history = updated_history

        with st.chat_message("assistant"):
            st.markdown(f"🤖 {response}")


if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_image = image.resize((800, 500))
            st.image(resized_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = get_image_caption(image)
            with col2:
                st.info(caption)
    else:
        st.warning("Please upload an image first.")


if selected == "Embed Text":
    st.title("🧠 Text Embedding with MY-GPT")

    input_text = st.text_area("Enter the text you want to embed:")

    if st.button("Generate Embedding"):
        if input_text.strip():
            with st.spinner("Generating embedding..."):
                embedding_vector = get_text_embedding(input_text)
            st.success("Embedding generated successfully!")

            st.write("📏 Embedding Vector Length:", len(embedding_vector))
            st.text_area("🔢 Embedding Vector (truncated)", value=str(embedding_vector[:10]) + " ...", height=150)
        else:
            st.warning("Please enter some text to embed.")