import pandas as pd
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load Data and Train Model
df = pd.read_csv('data/train.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Generate and Save the Plot Image
plt.switch_backend('Agg') 
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(X.iloc[y_kmeans == i, 0], X.iloc[y_kmeans == i, 1], 
                s=100, c=colors[i], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

if not os.path.exists('static'):
    os.makedirs('static')
plt.savefig('static/cluster_plot.png')
plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            income = float(request.form['income'])
            score = float(request.form['score'])
            
            cluster = kmeans.predict([[income, score]])[0]
            prediction_text = f"This customer belongs to Cluster {cluster}"
        except:
            prediction_text = "Error: Please enter valid numbers."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, port=5001)