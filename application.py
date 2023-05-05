import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application


model=pickle.load(open('models/model for carped.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler for carped.pkl','rb'))
# pca=pickle.load(open('models/pcapokemon.pkl','rb'))


@app.route('/')
def index():
    return 'hiii'

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        symboling=float(request.form.get('symboling'))
        carbody = float(request.form.get('carbody'))
        wheelbase = float(request.form.get('wheelbase'))
        carlength = float(request.form.get('carlength'))
        carwidth = float(request.form.get('carwidth'))
        carheight= float(request.form.get('carheight'))
        curbweight = float(request.form.get('curbweight'))
        
        enginetype = float(request.form.get('enginetype'))
        cylindernumber = float(request.form.get('cylindernumber'))
        enginesize = float(request.form.get('enginesize'))
        fuelsystem	 = float(request.form.get('fuelsystem'))
        boreratio = float(request.form.get('boreratio'))
        stroke = float(request.form.get('stroke'))
        compressionratio = float(request.form.get('compressionratio'))
        horsepower = float(request.form.get('horsepower'))
        peakrpm = float(request.form.get('peakrpm'))
        citympg = float(request.form.get('citympg'))
        highwaympg = float(request.form.get('highwaympg'))
        fueltype_diesel = float(request.form.get('fueltype_diesel'))
        fueltype_gas = float(request.form.get('fueltype_gas'))
        aspiration_std = float(request.form.get('aspiration_std'))

        aspiration_turbo = float(request.form.get('aspiration_turbo'))
        doornumber_four = float(request.form.get('doornumber_four'))
        doornumber_two = float(request.form.get('doornumber_two'))
        drivewheel_4wd = float(request.form.get('drivewheel_4wd'))
        drivewheel_fwd = float(request.form.get('drivewheel_fwd'))
        drivewheel_rwd = float(request.form.get('drivewheel_rwd'))
        enginelocation_front = float(request.form.get('enginelocation_front'))
        enginelocation_rear = float(request.form.get('enginelocation_rear'))


        
    
        new_data_scaled=standard_scaler.transform([[symboling, carbody, wheelbase, carlength, carwidth,carheight, curbweight, enginetype, cylindernumber, enginesize,fuelsystem, boreratio, stroke, compressionratio, horsepower,peakrpm, citympg, highwaympg, fueltype_diesel, fueltype_gas,aspiration_std, aspiration_turbo, doornumber_four,
        doornumber_two, drivewheel_4wd, drivewheel_fwd, drivewheel_rwd,enginelocation_front, enginelocation_rear]])
        # new_data_pca=pca.transform(new_data_scaled)
        result=model.predict(new_data_scaled)
        # if(result[0]==0):
        #     result='DrugY'
        # elif(result[0]==1):
        #     result='DrugC'
        # elif(result[0]==2):
        #     result='DrugX'
        # elif(result[0]==3):
        #     result='DrugA'
        # else:
        #     result='DrugB'
        return render_template('car pred.html',result=result)
    else:  
        return render_template('car pred.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
