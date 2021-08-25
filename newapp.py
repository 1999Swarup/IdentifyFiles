from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
from collections import Counter
import pickle
import time
from sklearn.preprocessing import StandardScaler
import psutil
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

model = pickle.load(open('modelfinal.sav', 'rb'))
ALLOWED_EXTENSIONS = {'jenkinsfile', 'csproj', 'rexx', 'ml', 'mak', 'kt'}
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    def probability_distribution(content):
        def _helper():
            absolute_distribution = Counter(content)
            length = len(content)
            for value, frequency in absolute_distribution.items():
                yield int(value), float(frequency) / length

        return Counter(dict(_helper()))

    # create a empty list, where you store the content
    list_of_text = []
    # create a empty list, where you store the content like values:frequency
    freq_value = []
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte
    # value occurs
    freq = []
    # create a empty list, where you store the byte values
    val = []
    # create a empty list, where you store the content like values:probability distribution - probability frequency
    # distribution value
    prob_freq_dist_value = []
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte
    # value occurs
    prob_freq = []
    # create a empty list, where you store the byte values
    prob_val = []
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c = probability_distribution(text)
    # List the n most common elements and their counts from the most common to the least.  If n is None,
    # then list all element counts.
    for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
        prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
        prob_val.append("{:02x}".format(prob_value))
        prob_freq.append("{:.0%}".format(prob_dist_freq))
    # List the n most common elements and their counts from the most common to the least.  If n is None,
    # then list all element counts.
    for value, frequency in c.most_common(n=5):
        freq_value.append("0x{:02x}: {}".format(value, frequency))
        val.append("{:02x}".format(value))
        freq.append("{}".format(frequency))
    list_of_text.append((val, freq, prob_freq, filename))
    df = pd.DataFrame(list_of_text,
                      columns=['OnlyValueForAbsoluteDistribution', 'OnlyFrequency', 'OnlyProbabilityDistribution',
                               'Filename'])

    df.reset_index(drop=True)
    # As all columns have the same number of lists, you can call Series.explode on each column
    df = df.set_index(['Filename']).apply(pd.Series.explode).reset_index()
    df.drop('Filename', axis=1, inplace=True)
    # removing the % sign present near data in the column probabilistic percentage
    df = df.replace('\%', '', regex=True)
    # removing all rows where the value in column "OnlyValueForAbsoluteDistribution" in non-numeric let's say stop words
    df = df[pd.to_numeric(df['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
    df['OnlyValueForAbsoluteDistribution'] = df['OnlyValueForAbsoluteDistribution'].astype(int)
    # replacing all infinite values with nan and than nan with mode value of that column
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['OnlyFrequency'].fillna(df['OnlyFrequency'].mode()[0], inplace=True)
    df['OnlyProbabilityDistribution'].fillna(df['OnlyProbabilityDistribution'].mode()[0], inplace=True)
    df['OnlyValueForAbsoluteDistribution'].fillna(df['OnlyValueForAbsoluteDistribution'].mode()[0], inplace=True)

    # changing data types of columns for each file type
    df['OnlyValueForAbsoluteDistribution'] = df.OnlyValueForAbsoluteDistribution.astype(int)
    df['OnlyFrequency'] = df.OnlyFrequency.astype(int)
    df['OnlyProbabilityDistribution'] = df.OnlyProbabilityDistribution.astype(int)

    # numerical features
    num_cols = ['OnlyValueForAbsoluteDistribution', 'OnlyFrequency', 'OnlyProbabilityDistribution']

    # apply standardization on numerical features
    for i in num_cols:
        scale = StandardScaler().fit(df[[i]])
        df[i] = scale.transform(df[[i]])

    df.reset_index(drop=True, inplace=True)

    start = time.time()

    listrow = []
    cjen = 0
    cml = 0
    cmak = 0
    ckt = 0
    ccs = 0
    crexx = 0
    # iterate through each row and select
    # 0th and 2nd index column respectively.
    for i in range(len(df)):
        listrow = []
        c1 = df.iloc[i, 0]
        c2 = df.iloc[i, 1]
        c3 = df.iloc[i, 2]
        listrow.append(c1)
        listrow.append(c2)
        listrow.append(c3)
        new_df = pd.DataFrame([listrow])
        res = model.predict(new_df)
        if res == 'jenkinsfile':
            cjen = cjen + 1
        if res == 'csproj':
            ccs = ccs + 1
        if res == 'rexx':
            crexx = crexx + 1
        if res == 'ml':
            cml = cml + 1
        if res == 'mak':
            cmak = cmak + 1
        if res == 'kt':
            ckt = ckt + 1

    end = time.time()
    eval_time = end - start

    cjenint = '{} %'.format((cjen / 5) * 100)
    ccsint = '{} %'.format((ccs / 5) * 100)
    crexxint = '{} %'.format((crexx / 5) * 100)
    cmlint = '{} %'.format((cml / 5) * 100)
    cmakint = '{} %'.format((cmak / 5) * 100)
    cktint = '{} %'.format((ckt / 5) * 100)
    cpuint = '{} %'.format(psutil.cpu_percent(interval=0.5))
    ramint = '{} %'.format(psutil.virtual_memory().percent)

    mystring = "jenkinsfile :  " + str(cjenint) + "\n csproj :  " + str(ccsint) + "\n rexx :  " + str(
        crexxint) + "\n  ml : " + str(cmlint) + "\n mak :  " + str(cmakint) + "\n kt :  " + str(
        cktint) + "\n CPU usage as a percentage :  " + str(
        cpuint) + "\n Number of predictions that can be made per sec :  " + str(
        1 / eval_time) + "\n RAM usage in percent :  " + str(ramint)

    return mystring


if __name__ == "__main__":
    app.run(debug=True)
