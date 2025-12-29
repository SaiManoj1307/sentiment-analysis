
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained emotion analysis pipeline
# "j-hartmann/emotion-english-distilroberta-base" detects: anger, disgust, fear, joy, neutral, sadness, surprise
print("Loading emotion model...")
sentiment_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
print("Model loaded!")

import cv2
import base64
from deepface import DeepFace

# Initialize - DeepFace loads models on first call or pre-load
# We will handle it in the request to be safe, or pre-warm if needed.

def get_spotify_suggestion(mood):
    mood = mood.lower()
    mood_map = {
        'joy': 'Happy Hits',
        'happy': 'Happy Hits',
        'sadness': 'Sad Songs',
        'sad': 'Sad Songs',
        'anger': 'Calm Vibes',
        'angry': 'Calm Vibes',
        'fear': 'Relaxing Music',
        'disgust': 'Feel Good',
        'surprise': 'Party Hits',
        'neutral': 'Deep Focus',
        'positive': 'Happy Upbeat',
        'negative': 'Mood Booster (Cheer Up)' # Kept for backward compatibility or potential future use
    }
    # Fallback
    query = mood_map.get(mood, 'Top Hits')
    return f"https://open.spotify.com/search/{query}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '')
        
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        # Max length for this specific model is typically 512 tokens
        results = sentiment_pipeline(review[:512]) 
        result = results[0]
        
        # Label is now distinct emotion: 'joy', 'sadness', 'anger', etc.
        label = result['label']
        score = result['score']
        
        suggestion = get_spotify_suggestion(label)

        return jsonify({
            'sentiment': label,
            'confidence': f"{score:.4f}",
            'review_snippet': review[:50] + "..." if len(review) > 50 else review,
            'spotify_link': suggestion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Read image
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        print(f"Analyzing image of shape: {img.shape}")

        # Analyze Emotions using DeepFace
        try:
            # enforce_detection=False prevents crash if no face is found
            analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        except Exception as e:
             print(f"DeepFace Error: {e}")
             return jsonify({'error': f"AI Model Error: {str(e)}"}), 500

        # Ensure analysis is a list
        if isinstance(analysis, dict):
            analysis = [analysis]

        faces_found = 0
        primary_emotion = "neutral" 
        max_face_area = 0

        for face in analysis:
            region = face.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            dominant_emotion = face.get('dominant_emotion', 'unknown')
            emotions = face.get('emotion', {})
            score = emotions.get(dominant_emotion, 0)
            
            # Determine primary emotion based on largest face
            area = w * h
            if area > max_face_area:
                max_face_area = area
                primary_emotion = dominant_emotion

            # Draw
            color = (0, 255, 0)
            if dominant_emotion in ['sad', 'angry', 'fear']:
                color = (0, 0, 255) # Red for negative
            
            # Make sure we don't draw on a 0-size rect
            if w > 0 and h > 0:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Label with background for readability
                label = f"{dominant_emotion.upper()}: {score:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw filled rect for text background
                cv2.rectangle(img, (x, y - 25), (x + tw + 10, y), color, -1)
                cv2.putText(img, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                faces_found += 1

        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        music_link = get_spotify_suggestion(primary_emotion)

        response_data = {
            'image_b64': img_str,
            'faces_found': faces_found,
            'spotify_link': music_link,
            'dominant_emotion': primary_emotion
        }
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
