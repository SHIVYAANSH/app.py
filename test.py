import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import google.generativeai as genai

# Configure the Gemini API with your API key
genai.configure(api_key="AIzaSyCWBxiwcigkVo0Xki9PuNVkGwDSK6LyAs4")

# Create the model configuration for Gemini API
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model for Gemini API
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Start the chat session for Gemini API
chat_session = model.start_chat(history=[])

# Initialize the BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate image description
def get_image_description(image):
    # Preprocess the image for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate the caption for the image
    out = captioning_model.generate(**inputs)
    
    # Decode the output to get the description
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# Streamlit app UI setup
st.title("Gemini AI with Image Description Integration")
st.subheader("Upload an image and ask your question:")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Input field for user question
user_input = st.text_input("Enter your question:", "")

# When image is uploaded
if uploaded_image:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the image description using BLIP model
    image_description = get_image_description(image)
    st.write("Image Description:")
    st.write(image_description)

    # If user has entered a question
    if user_input:
        # Combine image description and user input into a prompt for Gemini API
        prompt = f"Question: {user_input}\nImage Description: {image_description}"

        # Send the combined prompt to Gemini API
        response = chat_session.send_message(prompt)

        # Display the AI's response
        st.write("AI Response:")
        st.write(response.text)
