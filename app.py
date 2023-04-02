from flask import Flask, render_template, request
import pandas as pd
import pickle as pkl
import os

app = Flask(__name__)


@app.route('/')
def home():
    # orderDate = request.form['Order Date']
    # print(orderDate)
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        orderDate = request.form['Order Date']
        releaseDate = request.form['Release Date']
        brand = request.form['Brand'] 
        if brand == 'Yeezy': brand = ' Yeezy'
        sneakerName = request.form['Sneaker Name'].replace(' ','-')
        shoeSize = request.form['Shoe Size']
        buyerRegion = request.form['Buyer Region']
        retailPrice = request.form['Retail Price']

        orderDateSplit = orderDate.split('-')
        orderYear, orderMonth, orderDay = orderDateSplit[0], orderDateSplit[1], orderDateSplit[2]
        releaseDateSplit = releaseDate.split('-')
        releaseYear, releaseMonth, releaseDay = releaseDateSplit[0], releaseDateSplit[1], releaseDateSplit[2]


        brand_le = pkl.load(open(os.path.join('Model', 'brand_le.pkl'),'rb'))
        buyerRegion_le = pkl.load(open(os.path.join('Model', 'buyerRegion_le.pkl'),'rb'))
        shoeSize_le = pkl.load(open(os.path.join('Model', 'shoeSize_le.pkl'),'rb'))
        sneakerName_le = pkl.load(open(os.path.join('Model', 'sneakerName_le.pkl'),'rb'))
        regressor = pkl.load(open(os.path.join('Model', 'regressor.pkl'),'rb'))

        values = [orderDay, orderMonth, orderYear, releaseDay,
                  releaseMonth, releaseYear, brand_le.transform([brand])[0], sneakerName_le.transform([sneakerName])[0],
                  shoeSize_le.transform([float(shoeSize)])[0], buyerRegion_le.transform([buyerRegion])[0], retailPrice]
        columns = ['Order Day', 'Order Month', 'Order Year', 'Release Day',
                    'Release Month', 'Release Year', 'Brand', 'Sneaker Name', 
                    'Shoe Size', 'Buyer Region', 'Retail Price']
        inputRecord = pd.DataFrame(values).T
        inputRecord.columns = columns

        predict = regressor.predict(inputRecord)


    return render_template('predict.html', number=int(abs(predict)[0]))


if __name__=='__main__':
    app.run(debug=True)