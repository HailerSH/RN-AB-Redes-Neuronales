from django.shortcuts import render
from myapp.trained_model.main import main

def intro(request):
    """ Home page with project introduction """
    return render(request, 'myapp/intro.html')

def predict(request):
    """ Handles prediction form and returns scorecard """
    probabilities = None
    predictions = None

    if request.method == 'POST':
        income = float(request.POST.get('income'))
        age = int(request.POST.get('age'))
        loan_amount = float(request.POST.get('loan_amount'))

        probabilities, predictions = main(total_rec_prncp=0.1457, funded_amnt=0.1333, total_pymnt_inv=-0.2496)

    return render(request, 'myapp/predict.html', {'probabilities': probabilities, 'predictions': predictions})
