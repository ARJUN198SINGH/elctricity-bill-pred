import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application


model=pickle.load(open('models/model of drugs.pkl','rb'))
standard_scaler=pickle.load(open('models/standard drugs.pkl','rb'))
# pca=pickle.load(open('models/pcapokemon.pkl','rb'))


@app.route('/')
def index():
    return 'hiii'

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Age=float(request.form.get('Age'))
        Na_to_K = float(request.form.get('Na_to_K'))
        Sex_F = float(request.form.get('Sex_F'))
        Sex_M = float(request.form.get('Sex_M'))
        BP_HIGH = float(request.form.get('BP_HIGH'))
        BP_LOW= float(request.form.get('BP_LOW'))

        BP_NORMAL = float(request.form.get('BP_NORMAL'))
        Cholesterol_HIGH= float(request.form.get('Cholesterol_HIGH'))
        Cholesterol_NORMAL = float(request.form.get('Cholesterol_NORMAL'))
        # smoker_yes = float(request.form.get('smoker_yes'))
        # smoker_yes = float(request.form.get('smoker_yes'))
        # smoker_yes = float(request.form.get('smoker_yes'))
        # Speed	 = float(request.form.get('Speed'))
        
        # Generation	 = float(request.form.get('Generation'))
    
        new_data_scaled=standard_scaler.transform([ [Age,Na_to_K,Sex_F,Sex_M,BP_HIGH,BP_LOW,BP_NORMAL,Cholesterol_HIGH,Cholesterol_NORMAL]])
        # new_data_pca=pca.transform(new_data_scaled)
        result=model.predict(new_data_scaled)
        if(result[0]==0):
            result='DrugY'
        elif(result[0]==1):
            result='DrugC'
        elif(result[0]==2):
            result='DrugX'
        elif(result[0]==3):
            result='DrugA'
        else:
            result='DrugB'
        return render_template('drugs.html',result=result)
    else:  
        return render_template('drugs.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
