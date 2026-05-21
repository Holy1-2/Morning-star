import os
import json
import base64
import numpy as np
import cv2
import face_recognition
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "super_secret_face_key"
DB_FILE = "database.json"

# Helper: Load database
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, 'r') as f:
        return json.load(f)

# Helper: Save database
def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

# Helper: Convert Base64 Data URL from browser to an OpenCV/face_recognition image
def convert_base64_to_img(base64_string):
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert BGR (OpenCV) to RGB (face_recognition)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        images = data.get('images') # Array of 5 base64 string images

        db = load_db()
        if username in db:
            return jsonify({"status": "error", "message": "Username already exists."})

        encodings = []
        for img_str in images:
            img = convert_base64_to_img(img_str)
            # Find face locations and get encodings
            face_encs = face_recognition.face_encodings(img)
            if face_encs:
                encodings.append(face_encs[0].tolist()) # Convert numpy array to list for JSON

        if len(encodings) < 3: # Ensure we got at least a few clear facial reads
            return jsonify({"status": "error", "message": "Faces not clearly detected in photos. Try again."})

        # Store the average encoding or a list of encodings for that user
        db[username] = encodings
        save_db(db)
        return jsonify({"status": "success", "message": "Registration successful!"})

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        img_str = data.get('image')
        
        img = convert_base64_to_img(img_str)
        live_encodings = face_recognition.face_encodings(img)
        
        if not live_encodings:
            return jsonify({"status": "error", "message": "No face detected in camera view."})
        
        live_enc = live_encodings[0]
        db = load_db()
        
        # Loop through database to find a matching face
        for username, stored_encs in db.items():
            for stored_enc in stored_encs:
                # Compare live face to stored face
                match = face_recognition.compare_faces([np.array(stored_enc)], live_enc, tolerance=0.5)
                if match[0]:
                    session['user'] = username
                    return jsonify({"status": "success", "redirect": url_for('dashboard')})
                    
        return jsonify({"status": "error", "message": "Face recognition failed. Unknown User."})

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    db = load_db()
    all_users = list(db.keys()) # Get list of all authenticated personnel
    return render_template('dashboard.html', current_user=session['user'], users=all_users)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)