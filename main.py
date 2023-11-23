from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
import base64

app = Flask(__name__)

# Treinar e avaliar o modelo, cria uma instancia, treina, faz previsões e calcula metricas
def train_and_evaluate_model(classifier, X_train, X_test, y_train, y_test):
    model = classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    return acc, f1, cm


from sklearn.impute import SimpleImputer


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        classifier_name = request.form['classifier']

        new_data = pd.read_csv('car_price_prediction.csv',nrows=20000)

        selected_columns = ['Price', 'Mileage']
        X = new_data[selected_columns]

        # Tenta identificar dinamicamente a coluna de rótulos
        label_column = None
        for col in new_data.columns:
            if new_data[col].dtype == 'int64' or new_data[col].dtype == 'float64':
                continue # Ignora colunas numéricas
            if new_data[col].nunique() == new_data.shape[0]:
                continue # Ignora colunas com valores únicos
            label_column = col
            break

        if label_column is None:
            return "Não foi possível identificar uma coluna de rótulos no conjunto de dados."

        X = new_data.select_dtypes(include=['int64', 'float64'])
        y = new_data[label_column]

        # Tratar valores ausentes usando SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifiers = {
            'KNN': KNeighborsClassifier,
            'SVM': SVC,
            'Decision Tree': DecisionTreeClassifier,
            'Random Forest': RandomForestClassifier
        }

        # Seleciona o classificador com base na escolha do usuário
        classifier = classifiers[classifier_name]

        acc, f1, cm = train_and_evaluate_model(classifier, X_train, X_test, y_train, y_test)

        # Configura a figura e eixo para a matriz de confusão
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=200)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    if i == j:
                        color = 'blue'
                    else:
                        color = 'red'

                    ax.scatter(j, i, s=50, c=color, marker='o', edgecolors='black', linewidth=1, cmap=plt.cm.Blues)

        plt.xlabel('Quilometragem')
        plt.ylabel('Preço')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        img_str = base64.b64encode(buf.read()).decode('utf-8')

        return render_template('results.html', accuracy=acc, f1_score=f1, confusion_matrix=img_str)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
