from flask import Flask, request, jsonify, render_template
from gen import calculate_accuracyscore, getfeedback, getnextquestion
from werkzeug.utils import secure_filename
import os
from confidenceml import run_model
import requests
import google.generativeai as genai
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI API key
my_api_key = "AIzaSyCqNEgWQrilgTWFy7NKzlHTOQfVRCt-KKI"
genai.configure(api_key=my_api_key)

app = Flask(__name__)

# Initialize variables
prev_question = "What is Data Science?"
field = "Data Structures and Algorithm"
Score = 0
mul = 1
wrong_ans = 0

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# List of topics for the group discussion
topics = [
    "The impact of social media on society",
    "Climate change and its effects",
    "The future of artificial intelligence",
    "The importance of mental health awareness",
    "The role of technology in education",
    "The benefits and drawbacks of remote work",
    "The influence of pop culture on youth",
    "The ethics of genetic engineering",
    "The future of space exploration",
    "The importance of cybersecurity in the digital age"
]

# Select a random topic
chosen_topic = random.choice(topics)
print(f"Topic for discussion: {chosen_topic}")

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(user_input, context):
    try:
        response = model.generate_content(f"{context}\nUser: {user_input}\nGemini:")
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

def is_on_topic(user_input, topic):
    # Simple keyword check to see if the user's input is on topic
    topic_keywords = topic.lower().split()
    user_input_keywords = user_input.lower().split()
    common_keywords = set(topic_keywords) & set(user_input_keywords)
    return len(common_keywords) > 0

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/gd')
def gd():
    return render_template('gd.html')

@app.route('/mock_interview')
def mock_interview():
    return render_template('mock_interview.html', prev_question=prev_question, accuracy_score=0)  # Initialize accuracy_score to 0 or any default value

@app.route('/ask', methods=['POST'])
def ask_question():
    global prev_question, Score, mul, wrong_ans
    data = request.json
    answer = data.get('answer', '')
    if answer.lower() == "exit":
        return jsonify({"message": "Session ended", "final_score": Score})

    accuracy_score = calculate_accuracyscore(answer, prev_question)

    # Check if answer is correct
    Correct = True
    if accuracy_score < 30:
        Correct = False
        wrong_ans += 1
        mul = 1
    else:
        Score += accuracy_score * mul
        mul += 1

    # Ask for feedback on a wrong answer
    if accuracy_score < 60:
        feedback = getfeedback(prev_question, answer)
        return jsonify({"accuracy_score": accuracy_score, "feedback": feedback})

    # If answer is correct, get a new question based on the answer of the candidate
    prev_question = getnextquestion(prev_question, answer, Correct, field)
    return jsonify({"accuracy_score": accuracy_score, "next_question": prev_question})

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.json
    question = data.get('question', '')
    answer = data.get('answer', '')

    feedback = getfeedback(question, answer)
    return jsonify({"feedback": feedback})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    print("Received request to upload audio file.")
    
    if 'audio' not in request.files:
        print("No audio file provided in the request.")
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        print("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"File saved to {file_path}")

    # Run the ML model on the uploaded audio file
    confidence = run_model(file_path)
    print(f"Confidence score calculated: {confidence}")

    return jsonify({"confidence": confidence})

@app.route('/generate_gemini_response', methods=['POST'])
def generate_gemini_response():
    data = request.json
    user_input = data.get('user_input', '').strip().lower()
    topic = data.get('topic', chosen_topic)

    # Check if the user wants to end the discussion
    if user_input == "end":
        context = f"The group discussion on the topic: {topic} has ended. Please provide a constructive and analytical feedback on the user's participation. Highlight strengths and areas of improvement in a balanced manner. Keep it concise and professional."
        response = get_gemini_response(user_input, context)
        return jsonify({"response": response})

    # Check if the user's input is on topic
    if not is_on_topic(user_input, topic):
        return jsonify({"response": "Your response seems to be off-topic. Please stick to the topic."})

    # Set the context for argument-based discussion
    context = f"A group discussion is being held on the topic: {topic}. The user will speak first. You are supposed to argue with the user. Reply in 50 to 100 words only.If user writes end then give feedback about users participation."

    # Get Gemini's response that includes an argument
    response = get_gemini_response(user_input, context)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
