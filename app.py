from io import StringIO, BytesIO
from flask import Flask, url_for, render_template, request, redirect, session, send_file, jsonify
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from Bio import SeqIO
from Bio import AlignIO
import random
from config import mail_username, mail_password
import smtplib
import requests
import json
import zipfile
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import subprocess
import os.path
import pathlib
from collections import Counter
import csv
from Bio.Align.Applications import MuscleCommandline
from Bio.SeqRecord import SeqRecord
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from keras.models import load_model
import os
import base64
import pickle

warnings.filterwarnings("ignore")
plt.rcParams["figure.autolayout"] = True

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'webserver.iiserkol@gmail.com'
app.config['MAIL_PASSWORD'] = 'Satyam@iiserkol@webserver'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
db = SQLAlchemy(app)
mail = Mail(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    position = db.Column(db.String(100))
    institute = db.Column(db.String(100))
    email = db.Column(db.String(100))

    def __init__(self, username, password, position, institute, email):
        self.username = username
        self.password = password
        self.position = position
        self.institute = institute
        self.email = email


@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('dashboard.html', user=user)
    else:
        return render_template('index.html')


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        position = request.form['position']
        institute = request.form['institute']
        email = request.form['email']

        # Check if a user with the same username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('index.html', message="User Already Exists")

        # Create a new user and add it to the database
        new_user = User(username=username, password=password, position=position, institute=institute, email=email)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    else:
        return render_template('register.html')

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            session['username'] = data.username
            return redirect(url_for('index'))
        return render_template('index.html', message="Incorrect Details")

@app.route('/about/', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/theory_corner/', methods=['GET','POST'])
def theory_corner():
    return render_template('theory_corner.html')

@app.route('/tutorial', methods=['GET','POST'])
def tutorial():
    return render_template('tutorial.html')

@app.route('/contact_us', methods=['GET','POST'])
def contact_us():
    return render_template('contact_us.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    # Email server configuration
    smtp_server = 'smtp.mailgun.org'
    smtp_port = 587  # or the appropriate port for your email server
    smtp_username = 'postmaster@sandboxa1ed5b1d21d741efa784f54d1b3831cc.mailgun.org'
    smtp_password = 'Razor!@#$%^&*'
    sender_email = 'postmaster@sandboxa1ed5b1d21d741efa784f54d1b3831cc.mailgun.org'
    receiver_email = 'webserver.iiserkol@gmail.com'  # your email address

    # Compose the email message
    subject = f'New Contact Form Submission from {name}'
    body = f'Name: {name}\nEmail: {email}\nMessage: {message}'
    message = f'Subject: {subject}\n\n{body}'

    #Send the mail using Mailgun API
    response = requests.post(f'https://api.mailgun.net/v3/sandboxa1ed5b1d21d741efa784f54d1b3831cc.mailgun.org/messages',
                             auth=('api', '26943faf454ae07eb953f6cee35e5635-6d8d428c-a5fcaa3a'),
                             data={
                                 'from': sender_email,
                                 'to': receiver_email,
                                 'subject': subject,
                                 'text': body
                             })

    if response.status_code == 200:
        return jsonify({'success': True, 'message': 'Thank you for the submission'})
    else:
        return jsonify({'success': False, 'message': 'Oops! Something went wrong!!'})


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('dashboard.html', user=user)
    return render_template('index.html', message="Please login first")

@app.route('/preprocessing', methods=['GET','POST'])
def preprocessing():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('preprocessing.html', user=user)
    else:
        return redirect(url_for('login'))

def preprocess_sequences(records):
    # remove sequences with "X"
    print("Removing the duplicate sequences...")
    records = [r for r in records if "X" not in r.seq]

    # remove duplicate sequences
    sequences = []
    unique_records = []
    for record in records:
        sequence = str(record.seq)
        if sequence not in sequences:
            unique_records.append(record)
            sequences.append(sequence)
    print("Total number of sequences after removing duplicate sequences is:", len(unique_records))
    return unique_records

def filter_sequences(sequences, target_length):
    print("Removing unequal length sequences...")
    filtered_sequences = []
    for record in sequences:
        if len(record.seq) == target_length:
            filtered_sequences.append(record)
    print("Total number of sequences after making sequences of equal length is:", len(filtered_sequences))
    return filtered_sequences

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        user_length = request.form.get('length')
        uploaded_file = request.files['file']

        stringio = StringIO(uploaded_file.stream.read().decode("utf-8"))
        records = list(SeqIO.parse(stringio, "fasta"))
        print("Total number of sequences before preprocessing is ::", len(records))

        preprocessed_records = preprocess_sequences(records)
        sequence_filtered = filter_sequences(preprocessed_records, int(user_length))

        if(len(sequence_filtered)>0):
            with StringIO() as output:
                SeqIO.write(sequence_filtered, output, "fasta")
                processed_file = output.getvalue().encode()

        return send_file(BytesIO(processed_file), as_attachment=True, mimetype='application/octet-stream', download_name='processed.fasta')
    else:
        return render_template('index.html')

@app.route('/entropy', methods=['GET','POST'])
def entropy():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('entropy.html', user=user, num_files=1)
    else:
        return redirect(url_for('login'))

@app.route('/calculate_entropy', methods=['POST'])
def calculate_entropy():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        uploaded_files = []
        months = []
        years = []

        num_files = int(request.form.get('num_files'))

        for i in range(num_files):
            month = request.form.get(f'month_{i}')
            year = request.form.get(f'year_{i}')
            months.append(month)
            years.append(year)
            file = request.files[f'file_{i}']
            uploaded_files.append(file)

        def shannon_entropy(list_input):
            unique_aa = set(list_input)
            M = len(list_input)
            entropy_list = []

            for aa in unique_aa:
                n_i = list_input.count(aa)
                P_i = n_i / float(M)
                entropy_i = P_i * (math.log(P_i, 2))
                entropy_list.append(entropy_i)
            sh_entropy = -(sum(entropy_list))
            return sh_entropy

        def shannon_entropy_list_msa(alignment_file):
            shannon_entropy_list = []
            for col_no in range(len(list(alignment_file[0]))):
                list_input = list(alignment_file[:, col_no])
                shannon_entropy_list.append(shannon_entropy(list_input))
            return shannon_entropy_list

        entropy_plots = []
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, file in enumerate(uploaded_files):
                file_contents = file.read().decode("utf-8")
                stringio = StringIO(file_contents)
                with stringio:
                    align_clustal = AlignIO.read(stringio, "clustal")
                    clustal_omega = shannon_entropy_list_msa(align_clustal)

                    plt.plot(clustal_omega, label=f"{months[i]} {years[i]}")
                    plt.xlabel("Residue Position")
                    plt.ylabel("Shannon Entropy")
                    plt.legend()

                    entropy_plot_path = f"static/entropy_{months[i]}_{years[i]}.png"
                    plt.savefig(entropy_plot_path, dpi=300)
                    entropy_plots.append(entropy_plot_path)

                    plt.clf()

                    entropy_file = pd.DataFrame({"Residue Position": range(1, len(clustal_omega) + 1), "Shannon Entropy": clustal_omega})
                    entropy_csv_path = f"static/entropy_data_{months[i]}_{years[i]}.csv"
                    entropy_file.to_csv(entropy_csv_path, index=False)
                    zipf.write(entropy_csv_path)

        zip_buffer.seek(0)
        return render_template('entropy.html', user=user, num_files=num_files, entropy_plots=entropy_plots, zip_buffer=zip_buffer)
    else:
        return redirect(url_for('login'))

@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)

@app.route('/feature1', methods=['POST', 'GET'])
def feature1():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('feature1.html', user=user)
    else:
        return redirect(url_for('login'))

@app.route('/calculate_feature1', methods=['POST','GET'])
def calculate_feature1():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        max_pdp = []
        fcaa = []

        if uploaded_file.filename != '':
            stringio = StringIO(uploaded_file.stream.read().decode("utf-8"))
            sequence_file = list(SeqIO.parse(stringio, "fasta"))

            aa_pair = ['AA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AK', 'AL', 'AM', 'AN', 'AP', 'AQ', 'AR', 'AS','AT', 'AV', 'AW', 'AY',
                       'CA', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN', 'CP', 'CQ', 'CR', 'CS','CT', 'CV', 'CW', 'CY',
                       'DA', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DK', 'DL', 'DM', 'DN', 'DP', 'DQ', 'DR', 'DS','DT', 'DV', 'DW', 'DY',
                       'EA', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EK', 'EL', 'EM', 'EN', 'EP', 'EQ', 'ER', 'ES','ET', 'EV', 'EW', 'EY',
                       'FA', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FK', 'FL', 'FM', 'FN', 'FP', 'FQ', 'FR', 'FS','FT', 'FV', 'FW', 'FY',
                       'GA', 'GC', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GK', 'GL', 'GM', 'GN', 'GP', 'GQ', 'GR', 'GS','GT', 'GV', 'GW', 'GY',
                       'HA', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HK', 'HL', 'HM', 'HN', 'HP', 'HQ', 'HR', 'HS','HT', 'HV', 'HW', 'HY',
                       'IA', 'IC', 'ID', 'IE', 'IF', 'IG', 'IH', 'II', 'IK', 'IL', 'IM', 'IN', 'IP', 'IQ', 'IR', 'IS','IT', 'IV', 'IW', 'IY',
                       'KA', 'KC', 'KD', 'KE', 'KF', 'KG', 'KH', 'KI', 'KK', 'KL', 'KM', 'KN', 'KP', 'KQ', 'KR', 'KS','KT', 'KV', 'KW', 'KY',
                       'LA', 'LC', 'LD', 'LE', 'LF', 'LG', 'LH', 'LI', 'LK', 'LL', 'LM', 'LN', 'LP', 'LQ', 'LR', 'LS','LT', 'LV', 'LW', 'LY',
                       'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MI', 'MK', 'ML', 'MM', 'MN', 'MP', 'MQ', 'MR', 'MS','MT', 'MV', 'MW', 'MY',
                       'NA', 'NC', 'ND', 'NE', 'NF', 'NG', 'NH', 'NI', 'NK', 'NL', 'NM', 'NN', 'NP', 'NQ', 'NR', 'NS','NT', 'NV', 'NW', 'NY',
                       'PA', 'PC', 'PD', 'PE', 'PF', 'PG', 'PH', 'PI', 'PK', 'PL', 'PM', 'PN', 'PP', 'PQ', 'PR', 'PS','PT', 'PV', 'PW', 'PY',
                       'QA', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QK', 'QL', 'QM', 'QN', 'QP', 'QQ', 'QR', 'QS','QT', 'QV', 'QW', 'QY',
                       'RA', 'RC', 'RD', 'RE', 'RF', 'RG', 'RH', 'RI', 'RK', 'RL', 'RM', 'RN', 'RP','RQ', 'RR', 'RS', 'RT', 'RV', 'RW', 'RY',
                       'SA', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SK', 'SL', 'SM', 'SN', 'SP','SQ', 'SR', 'SS', 'ST', 'SV', 'SW', 'SY',
                       'TA', 'TC', 'TD', 'TE', 'TF', 'TG', 'TH', 'TI', 'TK', 'TL', 'TM', 'TN', 'TP','TQ', 'TR', 'TS', 'TT', 'TV', 'TW', 'TY',
                       'VA', 'VC', 'VD', 'VE', 'VF', 'VG', 'VH', 'VI', 'VK', 'VL', 'VM', 'VN', 'VP','VQ', 'VR', 'VS', 'VT', 'VV', 'VW', 'VY',
                       'WA', 'WC', 'WD', 'WE', 'WF', 'WG', 'WH', 'WI', 'WK', 'WL', 'WM', 'WN', 'WP','WQ', 'WR', 'WS', 'WT', 'WV', 'WW', 'WY',
                       'YA', 'YC', 'YD', 'YE', 'YF', 'YG', 'YH', 'YI', 'YK', 'YL', 'YM', 'YN', 'YP','YQ', 'YR', 'YS', 'YT', 'YV', 'YW', 'YY']

            for seq in sequence_file:
                aa = seq.seq
                aa_counts = []
                for pairs in aa_pair:
                    count = aa.count(pairs)
                    aa_counts.append(count)
                first = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W','Y']
                second = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W','Y']
                final = []

                for i in first:
                    for j in second:
                        x = aa.count(i) * aa.count(j) / len(aa)
                        final.append(x)

                diff = [x1 - x2 for (x1, x2) in zip(aa_counts, final)]

                rounded = []
                for a in diff:
                    r = round(a)
                    rounded.append(r)

                f1 = []
                for i in range(0, len(aa) - 2):
                    a = aa[i]
                    b = aa[i + 1]
                    c = aa[i + 2]
                    d = a + b
                    e = b + c
                    if (i == 0):
                        x1 = (aa.count(d) - round((aa.count(a) * aa.count(b)) / len(aa)))
                        f1.append(x1)
                    if (d, e in aa_pair):
                        x = (aa.count(d) - round((aa.count(a) * aa.count(b)) / len(aa))) + (
                                    aa.count(e) - round((aa.count(b) * aa.count(c)) / len(aa)))
                        f1.append(x)

                list1 = np.arange(0, len(aa) - 1)
                list2 = f1[0:len(aa) - 1]

            with zipfile.ZipFile('results.zip','w') as output_zip:
                for seq in sequence_file:
                    aa_dict1 = dict(zip(aa_list, max_pdp))
                    aa_dict2 = dict(zip(aa_list, fcaa))
                    csv_filename = seq.id + ".csv"
                    with open(csv_filename, "w") as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        writer.writerow(['Sequence', 'F1'])
                        for i in range(len(list1)):
                            writer.writerow([aa[i], list2[i]])

                    output_zip.write(csv_filename)
        if session.get('logged_in'):
            user = User.query.filter_by(username=session['username']).first()
            return render_template('feature1_1.html', user=user)
        else:
            return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route('/download1', methods=['GET'])
def download_results1():
    zip_file = 'results.zip'
    return send_file(zip_file, as_attachment=True)

@app.route('/feature2', methods=['POST', 'GET'])
def feature2():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('feature2.html', user=user)
    else:
        return redirect(url_for('login'))


def count_partitions(s, amino):
    # print("Total length of sequence is ::", len(s))
    # print("The number of A in sequence is ::", s.count(amino))

    partition = round((len(s) / s.count(amino)))
    g = []
    chunks, chunk_size = len(s), partition

    for i in range(0, chunks, chunk_size):
        x = s[i:i + chunk_size]
        g.append(x)

    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []

    for z in range(0, len(g)):
        b = g[z].count(amino)
        if (b == 0):
            count_0 += 1
            c0.append(count_0)
        elif (b == 1):
            count_1 += 1
            c1.append(count_1)
        elif (b == 2):
            count_2 += 1
            c2.append(count_2)
        elif (b == 3):
            count_3 += 1
            c3.append(count_3)
        elif (b == 4):
            count_4 += 1
            c4.append(count_4)
        elif (b == 5):
            count_5 += 1
            c5.append(count_5)

    # print("The partition containing 0", amino, "is ::", len(c0))
    # print("The partition containing 1", amino, "is ::", len(c1))
    # print("The partition containing 2", amino, "is ::", len(c2))
    # print("The partition containing 3", amino, "is ::", len(c3))
    # print("The partition containing 4", amino, "is ::", len(c4))
    # print("The partition containing 5", amino, "is ::", len(c5))

    def factorial(n):
        if (n == 1 or n == 0):
            return 1
        else:
            return n * factorial(n - 1)

    r = s.count(amino)
    n = len(g)
    first = factorial(r) / (
            factorial(len(c0)) * factorial(len(c1)) * factorial(len(c2)) * factorial(len(c3)) * factorial(
        len(c4)))
    second_den = []
    for i in range(0, len(g)):
        fact = factorial(g[i].count(amino))
        second_den.append(fact)

    product = 1
    for item in second_den:
        product = product * item
    second = factorial(r) / product
    third = n ** (-r)
    final = first * second * third
    return final


def calculate_max_proba(s, amino):
    # print("Total length of sequence is ::", len(s))
    # print("The number of A in sequence is ::", s.count(amino))
    aa_count = []
    new = s.count(amino)
    aa_count.append(new)

    def factorial(n):
        if (n == 1 or n == 0):
            return 1
        else:
            return n * factorial(n - 1)

    def distribution_probability(R, r, q, n):
        first_part = factorial(R)
        for qi in q:
            first_part = first_part / factorial(qi)

        second_part = factorial(R)
        for ri in r:
            second_part = second_part / factorial(ri)

        third_part = n ** (-R)

        return (first_part * second_part * third_part)

    def integer_partition(n):
        partitions = []

        def generate_partitions(n, max_val, current_partition):
            if (n == 0):
                partitions.append(current_partition)
            else:
                for i in range(1, min(n, max_val) + 1):
                    generate_partitions(n - i, i, current_partition + [i])

        generate_partitions(n, n, [])
        return partitions

    max_pdp = []

    for i in range(len(aa_count)):
        R = aa_count[i]
        n = aa_count[i]

        partitions = integer_partition(n)
        for p in partitions:
            while (len(p) < n):
                p.append(0)

        q_list = []
        for p in partitions:
            q_temp = []
            for i in range(len(p) + 1):
                qu = p.count(i)
                q_temp.append(qu)
            q_list.append(q_temp)

        final_dp = []
        for p, a in zip(partitions, q_list):
            r = [i for i in p]
            q = [i for i in a]
            prob = distribution_probability(R, r, q, n)
            final_dp.append(prob)
        max_pdp.append(max(final_dp))
    return max_pdp

@app.route('/calculate_feature2', methods=['POST'])
def calculate_feature2():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            with zipfile.ZipFile("output.zip", "w") as output_zip:
                stringio = StringIO(uploaded_file.read().decode("utf-8"))
                sequence_file = list(SeqIO.parse(stringio, "fasta"))
                aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

                for seq in sequence_file:
                    final_op = []
                    final_dp = []
                    for aa in aa_list:
                        count_op = count_partitions(seq, aa)
                        count_dp = calculate_max_proba(seq, aa)
                        final_op.append(count_op)
                        final_dp.append(count_dp)

                    final_value = []
                    for i in range(len(final_op)):
                        x = np.log(final_dp[i][0] / final_op[i])
                        final_value.append(x)

                    aa_dict1 = dict(zip(aa_list, final_value))
                    csv_filename = seq.id + ".csv"
                    with open(csv_filename, "w") as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        writer.writerow(['Sequence', 'F2'])
                        for residue in seq:
                            writer.writerow([residue, aa_dict1.get(residue)])

                    output_zip.write(csv_filename)
                    time.sleep(1)

            output_zip.close()

        if session.get('logged_in'):
            user = User.query.filter_by(username=session['username']).first()
            return render_template('feature2_1.html', user=user)
        else:
            return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route('/download2', methods=['GET'])
def download_results2():
    zip_file = 'results.zip'
    return send_file(zip_file, as_attachment=True)

@app.route('/feature3', methods=['POST','GET'])
def feature3():
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('feature3.html', user=user)
    else:
        return redirect('index.html')

@app.route('/calculate_feature3', methods=['POST','GET'])
def calculate_feature3():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        fcaa = []

        if uploaded_file.filename != '':
            stringio = StringIO(uploaded_file.stream.read().decode("utf-8"))
            sequence_file = list(SeqIO.parse(stringio, "fasta"))

            for seq in sequence_file:
                sequence = seq.seq
                A = sequence.count("A")
                C = sequence.count("C")
                D = sequence.count("D")
                E = sequence.count("E")
                F = sequence.count("F")
                G = sequence.count("G")
                H = sequence.count("H")
                I = sequence.count("I")
                K = sequence.count("K")
                L = sequence.count("L")
                M = sequence.count("M")
                N = sequence.count("N")
                P = sequence.count("P")
                Q = sequence.count("Q")
                R = sequence.count("R")
                S = sequence.count("S")
                T = sequence.count("T")
                V = sequence.count("V")
                W = sequence.count("W")
                Y = sequence.count("Y")

                A_cur = A / len(sequence)
                C_cur = C / len(sequence)
                D_cur = D / len(sequence)
                E_cur = E / len(sequence)
                F_cur = F / len(sequence)
                G_cur = G / len(sequence)
                H_cur = H / len(sequence)
                I_cur = I / len(sequence)
                K_cur = K / len(sequence)
                L_cur = L / len(sequence)
                M_cur = M / len(sequence)
                N_cur = N / len(sequence)
                P_cur = P / len(sequence)
                Q_cur = Q / len(sequence)
                R_cur = R / len(sequence)
                S_cur = S / len(sequence)
                T_cur = T / len(sequence)
                V_cur = V / len(sequence)
                W_cur = W / len(sequence)
                Y_cur = Y / len(sequence)

                A_fut = (((12 * A) + (2 * D) + (2 * E) + (4 * G) + (4 * P) + (4 * S) + (4 * T) + (4 * V)) / (36 * len(sequence)))
                C_fut = (((2 * R) + (2 * C) + (2 * G) + (2 * F) + (4 * S) + (2 * W) + (2 * Y)) / (18 * len(sequence)))
                D_fut = (((2 * A) + (2 * N) + (2 * D) + (4 * E) + (2 * G) + (2 * H) + (2 * Y) + (2 * V)) / (18 * len(sequence)))
                E_fut = (((2 * A) + (4 * D) + (2 * E) + (2 * Q) + (2 * G) + (2 * K) + (2 * V)) / (18 * len(sequence)))
                F_fut = (((2 * C) + (2 * I) + (6 * L) + (2 * F) + (2 * S) + (2 * Y) + (2 * V)) / (18 * len(sequence)))
                G_fut = (((4 * A) + (6 * R) + (2 * D) + (2 * C) + (2 * E) + (12 * G) + (2 * S) + (1 * W) + (4 * V)) / (36 * len(sequence)))
                H_fut = (((2 * R) + (2 * N) + (2 * D) + (4 * Q) + (2 * H) + (2 * L) + (2 * P) + (2 * Y)) / (18 * len(sequence)))
                I_fut = (((1 * R) + (2 * N) + (6 * I) + (4 * L) + (1 * K) + (3 * M) + (2 * F) + (2 * S) + (3 * T) + (3 * V)) / (27 * len(sequence)))
                K_fut = (((2 * R) + (4 * N) + (2 * E) + (2 * Q) + (1 * I) + (2 * K) + (1 * M) + (2 * T)) / (18 * len(sequence)))
                L_fut = (((4 * R) + (2 * Q) + (2 * H) + (4 * I) + (18 * L) + (2 * M) + (6 * F) + (4 * P) + (2 * S) + (1 * W) + (6 * V)) / (54 * len(sequence)))
                M_fut = (((1 * R) + (3 * I) + (2 * L) + (1 * K) + (1 * T) + (1 * V)) / (9 * len(sequence)))
                N_fut = (((2 * N) + (2 * D) + (2 * H) + (2 * I) + (4 * K) + (2 * S) + (2 * T) + (2 * Y)) / (18 * len(sequence)))
                P_fut = (((4 * A) + (4 * D) + (2 * Q) + (2 * H) + (4 * L) + (12 * P) + (4 * S) + (4 * T)) / (36 * len(sequence)))
                Q_fut = (((2 * R) + (2 * E) + (2 * Q) + (4 * H) + (2 * L) + (2 * K) + (2 * P)) / (18 * len(sequence)))
                R_fut = (((18 * R) + (2 * C) + (2 * Q) + (6 * G) + (2 * H) + (1 * I) + (4 * L) + (2 * K) + (1 * M) + (4 * P) + (6 * S) + (2 * T) + (2 * W)) / (54 * len(sequence)))
                S_fut = (((4 * A) + (6 * R) + (2 * N) + (4 * C) + (2 * G) + (2 * I) + (2 * L) + (2 * F) + (4 * P) + (14 * S) + (6 * T) + (1 * W) + (2 * Y)) / (54 * len(sequence)))
                T_fut = (((4 * A) + (2 * R) + (2 * N) + (3 * I) + (2 * K) + (1 * M) + (4 * P) + (6 * S) + (12 * T)) / (36 * len(sequence)))
                V_fut = (((4 * A) + (2 * D) + (2 * E) + (4 * G) + (3 * I) + (6 * L) + (1 * M) + (2 * F) + (12 * V)) / (36 * len(sequence)))
                W_fut = (((2 * R) + (2 * C) + (1 * G) + (1 * L) + (1 * S)) / (9 * len(sequence)))
                Y_fut = (((2 * N) + (2 * D) + (2 * C) + (2 * H) + (2 * F) + (2 * S) + (2 * Y)) / (18 * len(sequence)))

                aa_curr_count_list = [A_cur, C_cur, D_cur, E_cur, F_cur, G_cur, H_cur, I_cur, K_cur, L_cur,
                                      M_cur, N_cur, P_cur, Q_cur, R_cur, S_cur, T_cur, V_cur, W_cur, Y_cur]
                aa_fut_count_list = [A_fut, C_fut, D_fut, E_fut, F_fut, G_fut, H_fut, I_fut, K_fut, L_fut,
                                     M_fut, N_fut, P_fut, Q_fut, R_fut, S_fut, T_fut, V_fut, W_fut, Y_fut]

                fcaa = []
                for i in range(len(aa_curr_count_list)):
                    if (aa_curr_count_list[i] == 0):
                        ratio = 0
                        fcaa.append(ratio)
                    else:
                        ratio = aa_fut_count_list[i] / aa_curr_count_list[i]
                        fcaa.append(ratio)

                with zipfile.ZipFile('results3.zip', 'w') as output_zip:
                    aa_dict2 = dict(zip(aa_list, fcaa))
                    csv_filename = seq.id + ".csv"
                    with open(csv_filename, "w") as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        writer.writerow(['Sequence', 'F3'])
                        for residue in seq:
                            writer.writerow([residue, aa_dict2.get(residue)])
                    output_zip.write(csv_filename)

        if session.get('logged_in'):
            user = User.query.filter_by(username=session['username']).first()
            return render_template('feature3_1.html', user=user)
        else:
            return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route('/download3', methods=['POST','GET'])
def download_results3():
    zip_file = 'results3.zip'
    return send_file(zip_file, as_attachment=True)
    

def process_zip_file(zip_file, hidden_layers, neurons_per_layer, learning_rate, epochs, training_data_percentage):
    try:
        # Unzip the uploaded file in memory
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            files = zip_ref.infolist()
            results = {}

            for file in files:
                if file.filename.endswith('.csv'):
                    data = pd.read_csv(zip_ref.open(file))

                    # Assuming the model training and evaluation process here
                    X = data[['F1', 'F2', 'F3', 'F4']]
                    y = data['Target']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - training_data_percentage / 100, random_state=42)

                    model = MLPClassifier(hidden_layer_sizes=(neurons_per_layer,) * hidden_layers, learning_rate_init=learning_rate, max_iter=epochs)

                    # Creating a list to store the training accuracy and loss
                    training_accuracy = []
                    training_loss = []

                    for epoch in range(epochs):
                        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                        train_accuracy = model.score(X_train, y_train)
                        train_loss = model.loss_
                        training_accuracy.append(train_accuracy)
                        training_loss.append(train_loss)

                    # Collect the training history for plotting accuracy and loss curves
                    history = model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Model Evaluation
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)

                    roc_auc = roc_auc_score(y_test, y_pred)
                    fpr, tpr, _ = roc_curve(y_test, y_pred)

                    # Create a dictionary to store the model data and training history
                    model_data = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'recall': recall,
                        'precision': precision,
                        'roc_auc': roc_auc,
                        'training_accuracy': training_accuracy,
                        'training_loss': training_loss
                    }

                    # Create plots for ROC curves and training history
                    plt.figure(figsize=(3,2))
                    plt.plot(fpr, tpr, color='darkorange', lw=1)
                    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate', fontsize=5)
                    plt.ylabel('True Positive Rate', fontsize=5)
                    plt.title('ROC Curve', fontsize=5)
                    plt.xticks(fontsize=3)
                    plt.yticks(fontsize=3)
                    roc_curve_filename = f'{file.filename}_roc_curve.png'
                    plt.savefig(roc_curve_filename, dpi=300)

                    # Create accuracy and loss plot
                    plt.figure(figsize=(3,2))
                    plt.plot(training_accuracy, 'b', linewidth=1.0, label="Training Accuracy")
                    plt.plot(training_loss, 'r', linewidth=1.0, label="Training Loss")
                    plt.xlabel("Epochs", fontsize=5)
                    plt.ylabel("Training Percentage", fontsize=5)
                    plt.legend(fontsize=3   )
                    plt.title("Training Accuracy vs Loss", fontsize=5)
                    plt.xticks(fontsize=3)
                    plt.yticks(fontsize=3)
                    training_curve_filename = f'{file.filename}_accuracy_loss_curve.png'
                    plt.savefig(training_curve_filename, dpi=300)

                    # Convert plots to base64 strings for embedding in HTML
                    with open(roc_curve_filename, 'rb') as roc_file:
                        roc_curve_base64 = base64.b64encode(roc_file.read()).decode('utf-8')
                    with open(training_curve_filename, 'rb') as train_file:
                        train_curve_base64 = base64.b64encode(train_file.read()).decode('utf-8')

                    # Convert the model to bytes and then base64 string
                    model_file = BytesIO()
                    pd.to_pickle(model, model_file)
                    model_file.seek(0)
                    model_base64 = base64.b64encode(model_file.getvalue()).decode('utf-8')

                    # Add data to the results dictionary
                    results[file.filename] = {
                        'model_data': model_data,
                        'roc_curve': roc_curve_base64,
                        'model_file': model_base64,
                        'train_curve': train_curve_base64
                    }

        return results
    except Exception as e:
        print(e)
        return str(e)

@app.route('/model_train', methods=['POST', 'GET'])
def model_train():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            hidden_layers = int(request.form['hidden_layers'])
            neurons_per_layer = int(request.form['neurons_per_layer'])
            learning_rate = float(request.form['learning_rate'])
            epochs = int(request.form['epochs'])
            training_data_percentage = int(request.form['training_data_percentage'])

            results = process_zip_file(file, hidden_layers, neurons_per_layer, learning_rate, epochs, training_data_percentage)

            if isinstance(results, str):
                # If there was an error, return the error message as JSON
                return jsonify(error=results)

            # Return the results in JSON format
            return jsonify(results)
    if session.get('logged_in'):
        user = User.query.filter_by(username=session['username']).first()
        return render_template('model_train.html', user=user)
    else:
        return redirect(url_for('login'))

def model_predict(file, uploaded_model):
    try:
        results = {}

        data = pd.read_csv(file)

        x_test = data.drop('Target', axis=1)
        trained_model = pd.read_pickle(uploaded_model)
        predictions = trained_model.predict(x_test)

        plt.figure(figsize=(8,6))
        plt.plot(predictions)
        plt.xlabel("Residue")
        plt.ylabel("Mutational Probability")
        plt.title("Model Prediction")
        prediction_curve_filename = f'{file.filename}_prediction_curve.png'
        plt.savefig(prediction_curve_filename, dpi=300)

        with open(prediction_curve_filename, 'rb') as predict_curve:
            prediction_curve_base64 = base64.b64encode(predict_curve.read()).decode('utf-8')

        results[file.filename] = {
            'prediction_data': prediction_curve_base64
        }
        return results
    except Exception as e:
        print(e)
        return str(e)

@app.route('/model_test', methods=['POST', 'GET'])
def model_test():
    if request.method == 'POST':
        try:
            file = request.files['file']
            model = request.files['model_file']

            results = model_predict(file, model)

            return jsonify(results)

        except Exception as e:
            return jsonify(error=str(e))

    if 'logged_in' in session and session['logged_in']:
        user = User.query.filter_by(username=session['username']).first()
        return render_template('model_test.html', user=user)
    else:
        return redirect(url_for('login'))


if(__name__ == '__main__'):
    with app.app_context():
        app.secret_key = "ThisIsNotASecret:p"
        db.create_all()
    app.run(debug=True)