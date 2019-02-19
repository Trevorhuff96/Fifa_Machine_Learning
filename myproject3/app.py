from flask import Flask, request, jsonify, render_template
import numpy as np

from sklearn.externals import joblib





overall = joblib.load("features_to_overall.model")

wage = joblib.load("overall_to_wage.model")



app = Flask(__name__)





@app.route('/')

def index():

    return render_template('index.html')


@app.route("/overview.html")
def overview():
    return render_template("overview.html")


@app.route('/api/predict')

def predict():

    

    

    age   = request.args.get("age") or 0

    accel  = request.args.get("acceleration") or 0

    aggr = request.args.get("aggression") or 0

    ball_ctrl = request.args.get("ballControl") or 0

    dribble = request.args.get("dribbling") or 0

    finishing = request.args.get("finishing") or 0

    shot_pwr = request.args.get("shotPower") or 0

    sprint_speed = request.args.get("speed") or 0

    stamina = request.args.get("stamina") or 0

    strength = request.args.get("strength") or 0
    
    




    print("#"*80)

    features = [[age, accel, aggr, ball_ctrl, dribble, finishing, shot_pwr, sprint_speed, stamina, strength]]

    print(features)




    pred_overall = overall.predict(features)[0]
    print(pred_overall)
    overall_sqrd = pred_overall**2
    overall_sqroot = np.sqrt(pred_overall)
    overall_features =[[pred_overall, overall_sqrd, overall_sqroot]]
    print (overall_features)
    pred_wage = wage.predict(overall_features)

    #wage_prob= wage.predict_proba(pred_overall)



    print(pred_wage)


    print("#"*80)







    return jsonify([{"overall_pred": pred_overall.tolist(), 

                     "wage_pred": pred_wage.tolist(),
                     
                     "wage_prob": pred_wage.tolist(),

                     "features": features
    }])


if __name__ == "__main__":

    app.run(debug=True)