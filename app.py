import json, os

from src.answer_evaluation import *
from src.face_monitoring_inference import *
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
app.config['UPLOAD_IMAGE_FOLDER'] = 'store/images'
app.config['UPLOAD_AUDIO_FOLDER'] = 'store/audios'
app.config['UPLOAD_CV_FOLDER'] = 'store/cvs'
CORS(app)

@app.route('/api/face_detection', methods=['POST'])
def api_face_detection():
    username = request.form['username']
    image_file = request.files['image_file']

    save_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], secure_filename(image_file.filename)) 
    image_file.save(save_path)

    try:
        head_pose_text, det_username = face_image_inference(
                                                            username, 
                                                            save_path
                                                            )
    
        return Response(
                        response=json.dumps({
                                            "Head Pose": head_pose_text,
                                            "Username": det_username
                                            }),
                        status=200,
                        mimetype="application/json"
                        )
    
    except Exception as e:
        return Response(
                        response=json.dumps({
                                            "message": "Face detection failed",
                                            "error": str(e)
                                            }),
                        status=400,
                        mimetype="application/json"
                        )


    

    
@app.route('/api/answer_evaluation', methods=['POST'])
def api_answer_evaluation():
    data = request.form
    question = data['question']
    correct_answer = data['correct_answer']
    user_answer = data['user_answer']

    try:
        response = inference_answer_evaluation(question, correct_answer, user_answer)
    
        return Response(
                        response=json.dumps({
                                            "Score": response
                                            }),
                        status=200,
                        mimetype="application/json"
                        )
    
    except Exception as e:
        return Response(
                        response=json.dumps({
                                            "message": "Answer evaluation failed",
                                            "error": str(e)
                                            }),
                        status=400,
                        mimetype="application/json"
                        )
    

    
if __name__ == '__main__':
    app.run(
            debug=True, 
            host='0.0.0.0',
            port=5000
            )