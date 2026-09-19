# **ISOM5240** 

# Individual Assignment 

# **Storytelling Application using Hugging Face Pipelines** 

## **Objective:** 

This assignment assesses your Python programming skills and your ability to use Hugging Face's Transformers pipeline to build a storytelling application. The application should process an image provided by the user and generate a narrative. It must also be designed for deployment on Streamlit Cloud and should be suitable for use and play by children aged 3–10. 

## **Learning Outcomes:** 

1. Develop programming solutions to address business problems (PILO: 2, 3, 5, 7) 

2. Write computer programs using common programming practices (PILO: 2, 3, 5) 

3. Identify and fix logical and runtime errors in programs (PILO: 2, 3, 5) 

## **Assignment Requirements:** 

1. Develop a Python-based storytelling application using Hugging Face's Transformers pipeline. 

2. Image input: Users upload an image for the application to process. 

3. Story generation: The application generates a 50–100-word narrative based on details extracted from the uploaded image. 

4. Text-to-Speech Conversion: The generated story should be converted into an audio format to provide an engaging experience. 

5. Deployment: Deploy the final application on Streamlit Cloud so that users can interact with it online. 

## **Implementation Guidelines:** 

1. Image Processing & Captioning: 

   - Use a pre-trained image-captioning model from Hugging Face to generate a caption for the uploaded image. 

   - **Example** : The Salesforce/blip-image-captioning-base model may be used. 

2. Story Generation: 

   - Use a text-generation model to expand the caption into a complete story. 

## 3. Text-to-Speech Conversion: 

   - Use a TTS (text-to-speech) model from Hugging Face or Python's pyttsx3 or gTTS module to convert the generated text into speech. 

4. Deploy on Streamlit Cloud: 

   - Use Streamlit to create an interactive web UI. 

   - Deploy the application to Streamlit Cloud in accordance with its guidelines. 

## **Assessment Criteria:** 

|**Criterion**|**Description**|
|---|---|
|**Functionality**|The application correctly processes images, generates a story,<br>and converts the storyinto audio.|
|**Code Quality**|The code is well structured and documented.<br>The solution is implemented using**functions**to ensure<br>modularity and readability.<br>The code follows**common programming practices**, including<br>the use of meaningful variable names, proper indentation, and<br>code documentation.|
|**Model Usage**|Appropriate pre-trained models from Hugging Face are used<br>effectively.|
|**User Experience**|The application provides an interactive, user-friendly interface<br>usingStreamlit.|
|**Deployment**|The application is successfullydeployed on Streamlit Cloud.|



## **Submission Instructions:** 

- Submit the following files: 

   - app.py: Source code with comments 

   - requirements.txt: Dependencies 

   - Other files: Any other required files 

- Provide the Streamlit Cloud URL where the deployed application can be accessed. 

- Ensure that all functionality has been tested and documented. 

## **Late Submission Policy:** 

- Submissions up to 10 minutes late will incur a 30% penalty. 

- Submissions more than 1 hour late will not be accepted. 

