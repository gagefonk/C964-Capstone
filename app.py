import urllib
from flask import Flask, render_template, redirect, url_for, request
import pickle
import plotly
import plotly.graph_objects as go
from markupsafe import Markup
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.metrics import *
app = Flask(__name__)
# IMPORT CSV DATA
cvd = pd.read_csv('https://raw.githubusercontent.com/gagefonk/C964-Capstone/main/cv_disease.csv')
# IMPORT MACHINE LEARNING DATA
gs_classifier = pickle.load(urllib.request.urlopen("https://github.com/gagefonk/C964-Capstone/blob/68b1a65f6b108283dd441bb6c96c140c42118ae6/gs_classifier.pkl"))
pd.options.plotting.backend = 'plotly'
authenticated = False
pred_answers = []

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            authenticated = True
            return redirect(url_for('main'))
    return render_template('login.html', error=error)

@app.route('/main/', methods=['GET', 'POST'])
def main():
    checkAuth()
    pred_answers.clear()
    if request.method == 'POST':
        ageInput = request.form['ageInput']
        weightInput = request.form['weightInput']
        heighInput = request.form['heighInput']
        gender = request.form.get('gender')
        apHigh = request.form.get('apHigh')
        apLow = request.form.get('apLow')
        cholesterol = request.form.get('cholesterol')
        glucose = request.form.get('glucose')
        smoke = request.form.get('smoke')
        alcohol = request.form.get('alcohol')
        active = request.form.get('active')
        pred_answers.append(ageInput)
        pred_answers.append(weightInput)
        pred_answers.append(heighInput)
        if gender == 'Male':
            pred_answers.append(2)
        else:
            pred_answers.append(1)

        if apHigh == 'Normal':
            pred_answers.append(1)
        elif apHigh == 'Above Average':
            pred_answers.append(2)
        else:
            pred_answers.append(3)

        if apLow == 'Normal':
            pred_answers.append(1)
        elif apLow == 'Above Average':
            pred_answers.append(2)
        else:
            pred_answers.append(3)

        if cholesterol == 'Normal':
            pred_answers.append(1)
        elif cholesterol == 'Above Average':
            pred_answers.append(2)
        else:
            pred_answers.append(3)

        if glucose == 'Normal':
            pred_answers.append(1)
        elif glucose == 'Above Average':
            pred_answers.append(2)
        else:
            pred_answers.append(3)

        if smoke == 'No':
            pred_answers.append(0)
        else:
            pred_answers.append(1)

        if alcohol == 'No':
            pred_answers.append(0)
        else:
            pred_answers.append(1)

        if active == 'No':
            pred_answers.append(0)
        else:
            pred_answers.append(1)

        return redirect(url_for('predict'))
    return render_template('main.html')

@app.route('/predict/')
def predict():
    checkAuth()
    cvdResult, accuracy = make_prediction(pred_answers)
    return render_template('predict.html', cvdResult=cvdResult, accuracy=accuracy)

@app.route('/data/')
def data():
    checkAuth()
    bargraph = draw_fig_1()
    piecharts = draw_fig_2()
    heatmap = draw_fig_3()
    data = cvd.to_html(table_id="pandatable", max_rows=1000)
    return render_template('data.html', plot1=Markup(bargraph), plot2=Markup(piecharts), plot3=Markup(heatmap), df=Markup(data))

def checkAuth():
    if not authenticated:
        return redirect(url_for('login'))

#####FUNCTIONS######
# FIG 1
def draw_fig_1():
    age_male = cvd.loc[(cvd.GENDER == 2) & (cvd['CARDIO_DISEASE'] == 1)]
    age_female = cvd.loc[(cvd.GENDER == 1) & (cvd['CARDIO_DISEASE'] == 1)]
    trace1 = go.Histogram(
        x=age_male.AGE,
        opacity=0.75,
        name='Male',
        marker_color='blue')
    trace2 = go.Histogram(
        x=age_female.AGE,
        opacity=0.75,
        name='Female',
        marker_color='pink')
    layout = go.Layout(
        title='Male V Female With CVD',
        barmode='group',
        xaxis=dict(
            title='Age'
        ),
        yaxis=dict(
            title='Count'
        ),
        yaxis2=dict(
            title='Count',
            anchor='free',
            overlaying='y',
            side='right',
            position=1
        )
    )

    fig1 = go.Figure(data=[trace1, trace2], layout=layout)
    graph = plotly.offline.plot(fig1, output_type='div')
    return graph


# FIG 2
def draw_fig_2():
    cvd_present = cvd.loc[(cvd['CARDIO_DISEASE'] == 1)]
    smoke_data = cvd_present.SMOKE.value_counts()
    alcohol_data = cvd_present.ALCOHOL.value_counts()
    gluc_data = cvd_present.GLUCOSE.value_counts()
    chol_data = cvd_present.CHOLESTEROL.value_counts()

    colors = ['gold', 'mediumturquoise', 'darkorange']
    specs = [[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]
    fig2 = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=['Smoke', 'Alcohol', 'Glucose', 'Cholesterol'])
    fig2.add_trace(go.Pie(
        labels=['Doesn\'t Smoke', 'Smokes'],
        values=smoke_data.tolist(),
        name='Smoke Data'), 1, 1)
    fig2.add_trace(go.Pie(
        labels=['Doesn\'t Drink', 'Drinks'],
        values=alcohol_data.tolist(),
        name='Alcohol Data'), 1, 2)
    fig2.add_trace(go.Pie(
        labels=['Normal', 'Above Average', 'Very High'],
        values=gluc_data.tolist(),
        name='Glucose Data'), 2, 1)
    fig2.add_trace(go.Pie(
        labels=['Normal', 'Above Average', 'Very High'],
        values=chol_data.tolist(),
        name='Cholesterol Data'), 2, 2)

    fig2.update_traces(
        textfont_size=10,
        textinfo='label+percent',
        hole=.2,
        marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig2.update_annotations(yshift=30)
    fig2.update(layout_title_text='Statistics', layout_showlegend=False)

    graph = plotly.offline.plot(fig2, output_type='div')
    return graph


# FIG 3
def draw_fig_3():
    # GET CORRELATION
    corr = cvd.corr()
    heatmapobj = {'z': corr.values,
                  'x': corr.index.values,
                  'y': corr.columns.values}
    heatmap_fig = go.Figure(data=go.Heatmap(heatmapobj))
    heatmap_fig.update_layout(title='Correlation')
    graph = plotly.offline.plot(heatmap_fig, output_type='div')
    return graph

# MAKE PREDICTION
def make_prediction(ans):
    formatans = []
    formatans.append(ans)
    pred = gs_classifier.predict(formatans)
    acc = gs_classifier.predict_proba(formatans)
    if pred[0] == 0:
        predAns = 'False'
    else:
        predAns = 'True'

    accAns = '{:.2f}%'.format(acc[0][0] * 100)
    pred_answers.clear()
    return predAns, accAns

if __name__ == '__main__':
    app.run()