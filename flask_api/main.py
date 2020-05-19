from flask_cors import CORS
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import tensorflow as tf
from image_caption import predict_caption_for_image
# In[2]:

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app = Flask(__name__)
CORS(app)
api = Api(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
class MakePrediction(Resource):
    @staticmethod
    def post():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        data = request.files["file"]
        if data.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if data and allowed_file(data.filename):
            data.save("to_predict.jpg")

            img = tf.io.read_file("to_predict.jpg")
            img = tf.image.decode_jpeg(img, channels=3)
            result = predict_caption_for_image(img)

            return result

api.add_resource(MakePrediction, '/predict')


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
