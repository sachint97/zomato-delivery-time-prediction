from flask import Flask,request,render_template,jsonify
from src.utils import logging
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from datetime import datetime

application = Flask(__name__)
app=application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Delivery_person_Age=float(request.form.get('delivery_person_age')),
            Delivery_person_Ratings=float(request.form.get('delivery_person_ratings')),
            Restaurant_latitude=float(request.form.get('restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('restaurant_longitude')),
            Delivery_location_latitude=float(request.form.get('delivery_location_latitude')),
            Delivery_location_longitude=float(request.form.get('delivery_location_longitude')),
            Order_Date=datetime.strptime(request.form.get('order_date'), '%Y-%m-%d'),
            Time_Orderd=str(request.form.get('time_ordered')),
            Time_Order_picked=str(request.form.get('time_picked')),
            Weather_conditions=str(request.form.get('weather_conditions')).title(),
            Road_traffic_density=str(request.form.get('road_traffic_density')).title(),
            Vehicle_condition=float(request.form.get('vehicle_condition')),
            Type_of_order=str(request.form.get('type_of_order')).title(),
            Type_of_vehicle=str(request.form.get('type_of_vehicle')),
            multiple_deliveries=str(request.form.get('multiple_deliveries')).title(),
            Festival=str(request.form.get('festival')).title(),
            City=str(request.form.get('city')).title()        
        )
        final_new_data = data.get_data()
        logging.info(final_new_data.to_string())
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        results = round(pred[0],2)

        return render_template('result.html',final_result=results)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)