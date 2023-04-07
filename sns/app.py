from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask import make_response
from flask import render_template
import os
import SentenceProcessing
import random
import client
import pickle

app = Flask(__name__)
CORS(app)  #Allowing cross-domain requests ensures that the frontend can communicate with the backend

#Global variable
user_data = [None, None, None] #Record the prediction information entered by the user: [price/volume, model-name, days]
initial_state = 0 #Record which question the chatbot jumps to


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/message', methods=['POST'])
def message():
    global user_data
    global initial_state 
    data = request.get_json()
    user_message = data.get('text')
    chatbot_reply = ""
    image_url = None

    
    if initial_state == 0:
        chatbot_reply = "Which data would you like me to predict: the daily closing price or the daily trading volume?"
        initial_state = 1


    elif initial_state == 1: #question：price or volume?
        result = SentenceProcessing.object_detection(user_message)

        if result==1: # answer is price and volume
            chatbot_reply = "Please choose only one. Which one would you like me to predict: price or volume?"
        elif result==2: #answer is price
            user_data[0] = "price"
            chatbot_reply = "I will use the LSTM model for prediction. Would you like to change the model? (yes/no)"
            initial_state = 2
        elif result==3: #answer is volume
            user_data[0] = "volume"
            chatbot_reply = "I will use the LSTM model for prediction. Would you like to change the model? (yes/no)"
            initial_state = 2
        else: #wrong answer
            chatbot_reply = "Sorry, I didn't understand your response. Please adjust your answer."


    elif initial_state == 2: #question: whether change the model?
        result = SentenceProcessing.confirm_detection(user_message)
        if result == 1: #answer is yes
            chatbot_reply = "Which one of the following models would you like to choose: KNN / DT-entropy / DT-gini / GBDT / SVMLR / RF / RNN ?"
            initial_state = 3
        elif result == 2: #answer is no
            user_data[1] = 'lstm'
            chatbot_reply = "May I know for how many days starting from tomorrow you want me to predict? (Please enter an Arabic numeral less or equal to 7.)"
            initial_state = 4
        else: #wrong answer
            chatbot_reply = "Sorry, I didn't understand your response. Please answer with either yes or no."


    elif initial_state == 3: #question: which model to select?
        result = SentenceProcessing.mode_detection(user_message)
        if result: #answer is a correct model name
            user_data[1] = result
            chatbot_reply = "May I know for how many days starting from tomorrow you want me to predict? (Please enter an Arabic numeral less or equal to 7.)"
            initial_state = 4
        else: #wrong answer
            chatbot_reply = "Sorry, I didn't understand your response. Please select one of the following models: KNN / DT-entropy / DT-gini / GBDT / SVMLR / RF / RNN."


    elif initial_state == 4: #question: how many days to predict?
        result = SentenceProcessing.date_detection(user_message)
        if result == 8: #wrong answer (bigger than 7 or not an Arabic numeral)
            chatbot_reply = "Sorry, please enter an Arabic numeral less or equal to 7."
        else: #correct number
            user_data[2] = result
            #################################
            #################################

            final_code = SentenceProcessing.coding(user_data)
            feedback_list = client.client(final_code) #Invoke the prediction model and get the prediction results
            #feedback_list = pickle.load(feedback)

            #################################
            #################################
            chatbot_reply = f"The predictive data for the next {user_data[2]} days starting from tomorrow is: {feedback_list}."
            user_data = [None, None, None]
            initial_state = 0


    #response = make_response(jsonify({"reply": chatbot_reply, "image_url": image_url}))
    response = make_response(jsonify({"reply": chatbot_reply}))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/static/<path:path>')
def send_image(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
