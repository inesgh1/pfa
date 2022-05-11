from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)


@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    rest = request.form.get('restbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    rstg= request.form.get('rstg')
    thal = request.form.get('thal')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    th = request.form.get('th')
    input_query = np.array([[age,sex,cp,rest,chol,fbs,rstg,thal,exang,oldpeak,slope,ca,th]])
    result = model.predict(input_query)[0]
    return jsonify({'resultat':str(result)})
if __name__ == '__main__':
    app.run(debug=True)