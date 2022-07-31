import numpy as np
from flask import Flask,request,render_template
import pickle
process = pickle.load(open('processing.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    if request.method == 'POST':
        potential = float(request.form['potential'])
        heading_accuracy = float(request.form['heading_accuracy'])
        ball_control= float(request.form['ball_control'])
        reactions = float(request.form['reactions'])
        strength = float(request.form['strength'])
        interception = float(request.form['interception'])
        marking = float(request.form['marking'])
        sliding_tackle= float(request.form['sliding_tackle'])
        gk_diving = float(request.form['gk_diving'])
        gk_handling = float(request.form['gk_handling'])
        l = np.array([potential,heading_accuracy,ball_control,reactions,strength,interception,marking,sliding_tackle,gk_diving,gk_handling])
        p = l.reshape(1,-1)
        pro = process.transform(p)
        model_eval = model.predict(pro)[0]
        return  render_template('index.html',predection_eval = round(model_eval,2))

if __name__ == '__main__':
    app.run(debug=True)