#from calendar import c
from lib2to3.pgen2.token import NUMBER
import re
from tkinter import image_names
'''
#example
mystring = 'hello i want to pridict the price of amazon stock between 2023-4-4 and 2023-4-9'
numbers = re.findall(r'\d+\.\d+|\d+', mystring)
print(numbers)
'''

#These function is used to extract the keywords in one sentence
def date_detection(message):   
    #my_string = "hello i want to pridict the price of amazon stock between 2023/4/4 and 2023/4/9"
    numbers = []
    pattern = r"\d+"
    # Detect whether there is number in the senctence
    match = re.search(pattern, message)
    if match:
        current_number = ""
        for char in message:
            if char.isdigit() or char == "/":
                current_number += char
            elif current_number:
                numbers.append(current_number)
                current_number = ""
        if current_number:
            numbers.append(current_number)
        #print(numbers)
        #return numbers
        # convert the string type to int type
        numbers = abs(int(numbers[0]))
        # make sure the time lenght is smaller than 7
        if numbers > 7:
            # the data should be smaller than 7
            return 8
        else:
            # succeed, detected the numbers and it is smaller than 7
            return numbers
    else:
        # haven't detect the number
        return 8
    # Detect whether there is specific keyword in the sentence
    # Ignoring
def mode_detection(message):
    # Ignoring case sensitivity
    message = message.lower()
    if "lstm" in message:
        mode = "lstm"
    elif "rnn" in message:
        mode = "rnn"
    elif "knn" in message:
        mode = "knn"
    elif "dt-entropy" in message:
        mode = "dt-entropy"
    elif "dt-gini" in message:
        mode = "dt-gini"
    elif "gbdt" in message:
        mode = "gbdt"
    elif "svm" in message:
        mode = "svm"
    elif "lr" in message:
        mode = "lr"
    else:
         # Haven't detect any keywords, remind the user to choose the mode
        return False
    return mode
    
# detect the object the user want to predict, the close price or volume
def object_detection(message):
    message = message.lower()
    if "price" in message and "volume" in message :
        return 1
        # Tell the user to choose one of them to predict
        # This program is designed to predict one object a time
    elif "price" in message:
        # set the global variable as price
        return 2
    elif "volume" in message:
        # set the global variable as volume
        return 3
    else:
        return False
        # Haven't detect any keywords, remind the user to choose their purpose in corret form

def confirm_detection(message):
    message = message.lower()
    if "yes" in message and "no" in message:
        return 0
    elif "yes" in message:
        return 1
    elif "no" in message:
        return 2
    else:
        return 0

def coding(user_message):
    # this function will convert the elements in user_message list into an 16bits binary code
    # the first 2 bits represent the mode choose, there are 8 possible mode
    # the following 1 bit represent the object ,2 possible object: price or volume
    # the last 12 bits represent the time length, the user can predict maximum 2^12 = 4096 days of data in the future

    # user_message is a list
    # the 1st elements represents the object the user want to predict, price or volume
    # the 2nd elements represents the mode used in machine learning model
    # the 3rd elements represents length of time
    
    ##### mode #####
    if user_message[1] == "lstm":
        code1 = 0b000
    elif user_message[1] == "rnn":
        code1 = 0b001
    elif user_message[1] == "knn":
        code1 = 0b010
    elif user_message[1] == "dt-entropy":
        code1 = 0b011
    elif user_message[1] == "dt-gini":
        code1 = 0b100
    elif user_message[1] == "gbdt":
        code1 = 0b101
    elif user_message[1] == "svm":
        code1 = 0b110
    elif user_message[1] == "lr":
        code1 = 0b111
    
    ##### target #####
    if user_message[0] == "price":
        code2 = 0b0
    else:
        code2 = 0b1

    ##### timelength #####
    # convert the string type to int type
    code3 = int(user_message[2])

    
    final_code = (code1 << 13) | (code2 << 12) | code3  # combine the 3 codes
    # convert int type variable to string type variable
    final_code = bin(final_code)
    # delete the unecessary characters
    final_code = final_code[2:]
    # fill the code with zeros on the left side to make it a 16bits binary
    final_code = final_code.zfill(16)
    return final_code
# This function is used to decode the 16bits code sent by the client
def decoding(final_code):
    #user_message = []
    code1 = final_code[0:3]
    code2 = final_code[3]
    code3 = final_code[4:]

    #code1 mode
    ##### mode #####
    if code1 == '000':
        mode = 'lstm'
    elif code1 == '001':
        mode = 'rnn'
    elif code1 == '010':
        mode = 'knn'
    elif code1 == '011':
        mode = 'dt-entropy'
    elif code1 == '100':
        mode = 'dt-gini'
    elif code1 == '101':
        mode = 'gbdt'
    elif code1 == '110':
        mode = 'svm'
    elif code1 == '111':
        mode = 'lr'
    #default
    else:
        mode = 'lstm'
    ##### target #####
    if code2 == '1':
        target = 'volume'
    else:
        target = 'price'

    ##### timelength #####
    timelength = int(code3,2)

    return mode,target,timelength




