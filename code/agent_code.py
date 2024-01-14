import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import openai  # Add this import statement
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import datetime
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import re

openai.api_type = "azure"
openai.api_version = "2023-06-01-preview"
openai.api_base = "https://*******/"
openai.api_key = "*********"
os.environ['OPENAI_API_KEY'] = "***********"

st.markdown("""
    <style>
   
        .css-1aumxhk {
            width: 800px !important;
        }
        .css-1aumxhk textarea {
            background-color: #fff;
            color: #fff;
            font-size: 10px;
            font-style: italic;
        }
        
       bottom-space {
        margin-top: auto;
        }
        
       div.stButton > button {background-color: white; color: green;font-size: 12px;}
                
    </style>
""", unsafe_allow_html=True)

# Initialize input counter
input_counter = 0

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]
    st.session_state.conversation = ""
    st.session_state.qa = []

button_clicked = st.button("New chat", key="new_topics_button")
if button_clicked:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]
    st.session_state.conversation = ""
    st.session_state.qa = []
    st.session_state.data_result = ""

#loader = TextLoader(r"C:\streamlit_app\guide_data.txt")
#documents = loader.load()
#text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#chunks = text_splitter.split_documents(documents)
#vectordb = Chroma.from_documents(
#        documents=chunks,
#         embedding=OpenAIEmbeddings(deployment_id='textembeddingada002'),
#        persist_directory=r"C:\streamlit_app\data_st\chroma_db")

def get_retriever():
    embeddings_model = OpenAIEmbeddings(deployment_id='textembeddingada002')
    loaded_vectordb = Chroma(persist_directory=r"C:\streamlit_app\data_st\chroma_db", embedding_function=embeddings_model)
    return loaded_vectordb.as_retriever()

def remove_comments(code):
    cleaned_code = re.sub(r'#.*', '', code)
    return cleaned_code

def answer_question(question, history):
    
    # question = " Retrieve information in the form of a DataFrame, which includes the count and percentage of missing values for each column across all rows in the dataset"
    # question = " Inform me about the business context related to the data and the data analysis you are performing"
    # question = " provide me the shape: row and column count of the data "
    # question = " Provide a DataFrame displaying the monthly trend (in the year 2010) of the sum of Quantity for the Description 'SET/6 STRAWBERRY PAPER PLATES'. The resulting DataFrame should have two columns: the first one indicating the month (1, 2, 3, etc.) in the year 2010, and the second one representing the sum of Quantity grouped by month."
    # question = Generate a bar chart where the x-axis represents the "Description" and the y-axis represents the total quantity (sum of Quantity) for each Description. Display the top 4 categories based on the highest sum of Quantity"
    # question = "Kindly construct a repurchase predictive model and execute the model. "
    # question = "do not build model, only do data analysis: provide me the data frame that contains the data types of the all the columns in the data set"
    
    #history = "role: assistant, content: 'Hi, how can I help you?'"
    
    file_path = 'C:/streamlit_app/repeat_model.py'
    with open(file_path, 'r') as file:
        rep_model_instruction = file.read()
            
    file_path = 'C:/streamlit_app/AI_prompt.txt'
    with open(file_path, 'r') as f:
        promtsv = f.read()        
   
    exec(promtsv, globals())
        
    mpnms_lst = st.session_state.modelpara_names.split(',')
    
    st.session_state.template = st.session_state.template +  st.session_state.datastory + st.session_state.hint
        
    prompt = ChatPromptTemplate.from_template(st.session_state.template)
    
    llm = AzureOpenAI(deployment_id='gpt4', model_name="gpt-4")
    retriever = get_retriever()
    
    question_cp = question[:]
        
    question = st.session_state.question_hint + st.session_state.specification + st.session_state.modelpara_hint + ', now my question is:'+ question
    
    L = len(st.session_state.qa)
    
    if question_cp == 'Yes' and L>0:
        previous_qa = st.session_state.qa[L-1]
        previous_q_res = 'question:' + previous_qa[0] + ', here is the result for this question:' + st.session_state.data_result
        st.session_state.question_hint =  st.session_state.question_hintyes
        st.session_state.specification  =  st.session_state.specificationyes
        question = st.session_state.question_hint + previous_q_res + st.session_state.specification 
        
    if question_cp == 'No' and L>0:
        question = st.session_state.question_hintno
            
    if any(column in question_cp for column in mpnms_lst):
       question = "From the followig text string:" + " '" + question_cp + "'"
       question = question + st.session_state.get_modelpara + st.session_state.default_model_para
            
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )
        
    response = rag_chain.invoke(question)    
    
    print ("-----------respnse----------------------------------")
    print (response)
    
    response_pythob_code = ""
    model_result = ""
                  
    if any(column in response for column in mpnms_lst):
      
        python_code_match = re.search(r"```python\n(.+?)\n```", response, flags=re.DOTALL)
        if python_code_match:
            st.session_state.model_para_instruction = python_code_match.group(1)   
            st.session_state.model_para_instruction = st.session_state.model_para_instruction.replace(';', '')            
            response = "#modeling script in Python:[run]"
            response = response + '\n' + st.session_state.importread +\
                       '\n' + st.session_state.default_model_para + '\n' + \
                        st.session_state.model_para_instruction + '\n' + rep_model_instruction
            response_pythob_code = remove_comments(response)
            print (response)
            response_pythob_code = response_pythob_code.replace('\n', '<br>')
                   
    if st.session_state.modelconfirm in response:
        response = st.session_state.modelconfirm + ":"
        response = response + '\n' + st.session_state.model_para_instruction 
                    
    if  ("[run]" in response) == False:
                
        python_code_match = re.search(r"```python\n(.+?)\n```", response, flags=re.DOTALL)
        
        just_code = 0
        if python_code_match:
            pythoncode = python_code_match.group(1)
            just_code = 1
        else:
            just_code = 0
            print("No Python code found.")
        
        if just_code == 1:     
            existcode_flag = 0    
            try: 
                if not df.empty:
                    existcode_flag = 1
                else:
                    existcode_flag = 0
            except:
                existcode_flag = 0            
                
            if existcode_flag == 0:
                pythoncode =  st.session_state.importread + '\n' + pythoncode
                
            exec(pythoncode, globals())
            
            response = st.session_state.suggestion
            
    if ("[run]" in response):
        exec(response, globals())
        model_result = 'auc_score:' + str(auc_score) + '   ks_statistic:' + str(ks_statistic)  
        response = 'The modeling result: ' + "\n" + \
        model_result + "\n" + st.session_state.suggestion
    
    st.session_state.data_result = ""    
    
    try: 
        type(result)
        table_placeholder = st.sidebar.empty()
        if str(type(result)) != "<class 'pandas.core.frame.DataFrame'>":
            x_string = 'The answer: ' + str(result)
            table_content = f"<div style='text-align: left;'>{x_string}</div>"
            table_placeholder.markdown(table_content, unsafe_allow_html=True)
            st.session_state.data_result = x_string
        else:    
            table_style = [
            dict(selector="th", props=[("background-color", "lightgray"), ("color", "black"), ("font-size", "11px"), ("text-align", "center")]),
            dict(selector="td", props=[("background-color", "white"), ("color", "black"), ("font-size", "11px"), ("text-align", "left")]),
            dict(selector="tr:hover", props=[("background-color", "lightgray")]),
            ]
            table_placeholder.table(result.style.set_table_styles(table_style))           
            st.session_state.data_result = str(result.to_string())
    except:
        pass
                
    return [response, st.session_state.data_result, response_pythob_code]

# React to user input
if prompt := st.chat_input(f"Write here.."):
    promptv = prompt.replace(';', '\n')            
    st.chat_message("user").markdown(promptv)
    prompt = prompt.replace(';', '')            
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    history = " ".join([f""" "Role":"{msg["role"]}" \n "Content": "{msg["content"]}" """ for msg in st.session_state.messages])
    
    # Increment the input counter
    input_counter += 1

    # Bot answer
    with st.chat_message("assistant"):
        # Get bot answer
        answer, data_res, response_pythob_code = answer_question(
            question=prompt,
            history=history
        )
        
        styled_text = f'<span style="color: rgb(48, 129, 40);">{answer}</span>'
        st.markdown(styled_text, unsafe_allow_html=True)
    
        
    if len(response_pythob_code)>1:
        response_pythob_code = response_pythob_code
    else:    
        response_pythob_code = " "
    
    prompt = '<br>' + prompt.replace('\n', '<br>')    
    answer = '<br>' + answer
    q_and_a = (
    f'<span style="color: black; background-color: white;"><b>[user]:</b>&nbsp; {prompt}</span> <br>'
    f'<span style="color: rgb(48, 129, 40); background-color: white;"><b>[assistant]:</b>&nbsp; {answer}</span>'
    )
    
    ll = '<br><br>'
    st.session_state.conversation = st.session_state.conversation + "<br>" + ll + q_and_a
    
    if len(data_res)> 0:
        answer = answer + ' here is the result after running Python code: ' + data_res
        
    st.session_state.messages.append({"role": "assistant", "content": answer})    
    st.session_state.qa.append([prompt, answer])


table_placeholder = st.sidebar.empty()

message_ini = "Data Results Will be Here"
table_content = f"<div style='text-align: left; color: #3f3e3e42; font-size: 25px;'>{message_ini}</div>"
table_placeholder.markdown(table_content, unsafe_allow_html=True)

readonly_styles = """
    <style>
        .readonly-text-area {
            color: black;
            background-color: white;
            font-size: 14px;
            font-style: italic;
            border: 1px solid #ccc;
            padding: 8px;
            overflow-y: auto;
        }
    </style>
"""

# Display the readonly text area using st.markdown
if st.session_state.conversation != "":
    st.markdown(readonly_styles + f'<div class="readonly-text-area">{st.session_state.conversation}</div>', unsafe_allow_html=True)
