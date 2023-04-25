from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('home.html')
    else:
        df = CustomData(
            request.form.get('gender'),
            request.form.get('ethnicity'),
            request.form.get('parental_level_of_education'),
            request.form.get('lunch'),
            request.form.get('test_preparation_course'),
            int(request.form.get('writing_score')),
            int(request.form.get('reading_score'))
        ).getData_as_DataFrame()

        result = PredictPipeline().predict(df)
        return render_template('home.html',results=result[0])

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)