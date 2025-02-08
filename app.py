from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import time

app = Flask(__name__)

# Load model
model = joblib.load('D:\Desktop\colorectalcancermodelonline\colorectal.joblib')
feature_columns = ['CEA', 'ALB', 'CIKP', 'Cyfra211', 'Ca', 'HGB']

# Configure matplotlib for non-GUI backend
plt.switch_backend('Agg')

def create_shap_plot(input_data):
    # Create explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Generate plot
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0][0, :],
            base_values=explainer.expected_value[0],
            data=input_data.values[0],
            feature_names=feature_columns
        ),
        max_display=6
    )
    
    # Save plot
    plot_path = os.path.join('static', 'shap_plots', f'shap_{int(time.time())}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = {
            'CEA': float(request.form['CEA']),
            'ALB': float(request.form['ALB']),
            'CIKP': float(request.form['CIKP']),
            'Cyfra211': float(request.form['Cyfra211']),
            'Ca': float(request.form['Ca']),
            'HGB': float(request.form['HGB'])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([data], columns=feature_columns)
        
        # Make prediction
        proba = model.predict_proba(input_df)[0]
        
        # Generate SHAP plot
        shap_plot = create_shap_plot(input_df)
        
        return jsonify({
            'cancer_prob': round(proba[1]*100, 2),
            'polyp_prob': round(proba[0]*100, 2),
            'shap_plot': shap_plot
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


def generate_shap_plot(input_data, prediction_type):
    X_test = pd.DataFrame([input_data], columns=feature_columns)

    # 获取模型类别信息
    class_names = model.classes_  # 新增
    print(f"Model classes: {class_names}")  # 调试信息
    

    # 确定类别索引
    class_index = 0 if prediction_type == 'cancer' else 1  # 修改这里
    print(f"Requested class index: {class_index}")  # 调试信息
    
    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    

    # 调试输出SHAP值结构
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values length: {len(shap_values)}")
        for i, v in enumerate(shap_values):
            print(f"SHAP values[{i}] shape: {v.shape}")
    
    # 处理单类别输出的情况
    if len(shap_values) == 1:
        shap_values = shap_values[0]
    
    # 获取样本值（添加异常处理）
    try:
        shap_values_sample = shap_values[class_index].T[0]
    except IndexError:
        print("Using alternative SHAP values indexing")
        shap_values_sample = shap_values.T[0]  # 备用索引方式

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'shap_plots'), exist_ok=True)
    app.run(debug=True)

