from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk halaman Accuracy
@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

# Rute untuk halaman Kontak
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Rute untuk halaman Dataset (misalnya)
@app.route('/dataset')
def dataset():
    # Baca file CSV
    df = pd.read_csv('./dataset/Brain_Tumor.csv')

    # Ambil kolom dan data
    columns = df.columns.tolist()
    data = df.to_dict(orient='records')

    # Pastikan data tidak None
    if data is None:
        data = []
    if columns is None:
        columns = []

    # Render template dengan data
    return render_template('dataset.html', columns=columns, data=data)

if __name__ == '__main__':
    app.run(debug=True)
