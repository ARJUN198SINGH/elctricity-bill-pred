import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application


model=pickle.load(open('models/model of elicticity bill.pkl','rb'))
standard_scaler=pickle.load(open('models/scaling of elicticity bill.pkl','rb'))
# pca=pickle.load(open('models/pcapokemon.pkl','rb'))


@app.route('/')
def index():
    return 'hiii'

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        num_rooms=float(request.form.get('num_rooms'))
        num_people = float(request.form.get('num_people'))
        housearea = float(request.form.get('housearea'))
        is_ac = float(request.form.get('is_ac'))
        is_tv = float(request.form.get('is_tv'))
        is_flat= float(request.form.get('is_flat'))
        ave_monthly_income = float(request.form.get('ave_monthly_income'))
        num_children= float(request.form.get('num_children'))
        is_urban = float(request.form.get('is_urban'))
       

        # stroke = float(request.form.get('stroke'))
        # compressionratio = float(request.form.get('compressionratio'))
        # horsepower = float(request.form.get('horsepower'))
        # peakrpm = float(request.form.get('peakrpm'))
        # citympg = float(request.form.get('citympg'))
        # highwaympg = float(request.form.get('highwaympg'))
        # fueltype_diesel = float(request.form.get('fueltype_diesel'))
        # fueltype_gas = float(request.form.get('fueltype_gas'))
        # aspiration_std = float(request.form.get('aspiration_std'))

        # aspiration_turbo = float(request.form.get('aspiration_turbo'))
        # doornumber_four = float(request.form.get('doornumber_four'))
        # doornumber_two = float(request.form.get('doornumber_two'))
        # drivewheel_4wd = float(request.form.get('drivewheel_4wd'))
        # drivewheel_fwd = float(request.form.get('drivewheel_fwd'))
        # drivewheel_rwd = float(request.form.get('drivewheel_rwd'))
        # enginelocation_front = float(request.form.get('enginelocation_front'))
        # enginelocation_rear = float(request.form.get('enginelocation_rear'))


        q=np.array([[num_rooms, num_people, housearea, is_ac, is_tv, is_flat,ave_monthly_income, num_children, is_urban]])
        new_data_scaled=standard_scaler.transform(q)
      
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
        return render_template('Household energy bill data.html',result=np.round(result,2))
    else:  
        return render_template('Household energy bill data.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
