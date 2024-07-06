from django.shortcuts import render

# our home page view
def home(request):    
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare):
    import pickle
    model = pickle.load(open("C:\\Users\\Mukund Agarwalla\\Desktop\\Titanic_Survial_Prediction\\Titanic_Survial_Prediction\\Titanic_prediction\\titanic_survival_ml_model.sav", "rb"))

    scaled = pickle.load(open("C:\\Users\\Mukund Agarwalla\\Desktop\\Titanic_Survial_Prediction\\Titanic_Survial_Prediction\\Titanic_prediction\\scaler.sav", "rb"))

    pclass = int(pclass)
    sex = int(sex)
    age = int(age)
    sibsp = int(sibsp)
    parch = int(parch)
    fare = float(fare)
    

    scaled_input = scaled.transform([[pclass, sex, age, sibsp, parch, fare]])
    prediction = model.predict(scaled_input)
    
    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"
        

# our result page view
def result(request):
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])


    result = getPredictions(pclass, sex, age, sibsp, parch, fare)

    return render(request, 'result.html', {'result':result})