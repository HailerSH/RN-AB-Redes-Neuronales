import pandas as pd
from .preproccess import preprocess_input
from .usoModelo import load_model, predict
import os

def main(total_rec_prncp: float, funded_amnt: float, total_pymnt_inv: float) -> tuple:
    filesPath = "."
    selected_model = "3_mas_importantes"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    min_vals = pd.read_csv(os.path.join(BASE_DIR, "Scaler", "min_values.csv"), index_col=0).squeeze()
    max_vals = pd.read_csv(os.path.join(BASE_DIR, "Scaler", "max_values.csv"), index_col=0).squeeze()
    expected_features = list(pd.read_csv(os.path.join(BASE_DIR, "SelectedColumns", f"{selected_model}.csv"))['column_name'])
    categorical_columns = []  # Del formulario se escriben las que sean categoricas

    input = pd.DataFrame([{
        "total_rec_prncp": total_rec_prncp,  
        "funded_amnt": funded_amnt, 
        "total_pymnt_inv": total_pymnt_inv
    }]) ## acá se pone la información del formulario

    processed = preprocess_input(input,  min_vals, max_vals, categorical_columns, expected_features)

    model, device = load_model(path=os.path.join(BASE_DIR, "Models", f"best_model_for_{selected_model}.pth"), 
                               input_size=3)

    predictionDict = predict(processed, model, device)

    return predictionDict["probabilities"], predictionDict["predictions"]
