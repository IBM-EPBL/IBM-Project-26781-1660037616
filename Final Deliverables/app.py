import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import os
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
from flask import Flask
from turtle import st
from flask import Flask, render_template, request, redirect, url_for, session
from markupsafe import escape
import ibm_db

conn = ibm_db.connect("DATABASE=bludb;HOSTNAME=1bbf73c5-d84a-4bb0-85b9-ab1a4348f4a4.c3n41cmd0nqnrk39u98g.databases.appdomain.cloud;PORT=32286;SECURITY=SSL;SSLServerCertificate=DigiCertGlobalRootCA.crt;UID=nmm36298;PWD=zmbdjcoUXumahPoj",'','')

app = Flask(__name__)

picsFolder = os.path.join('static','pics')
app.config['UPLOAD_FOLDER'] = picsFolder
var_list = []

@app.route('/')
def home():
    img = os.path.join(app.config['UPLOAD_FOLDER'],'diabetic-retinopathy-illustration_42265-59.webp')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.jpg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmlogin.jpg')
    return render_template('ibmprojecthome.html',imgg=img,imgg1=img1,imgg2=img2)

@app.route('/register')
def register():
    img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
    return render_template('ibmprojectregister.html',imgg3=img3)

@app.route('/addrec',methods=['POST','GET'])
def addrec():
    if request.method == 'POST' :
        NAME = request.form['NAME']
        EMAIL = request.form['EMAIL']
        PASSWORD = request.form['PASSWORD']
        sql = "SELECT * FROM Users WHERE NAME =?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt,1,NAME)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        if account:
            msg = 'Account already exists !'
        
        else:

            var_list.append(NAME)
            var_list.append(EMAIL)
            var_list.append(PASSWORD)
            bodytemp = r"D:\\ibmme\\templates\\email.html"
            with open(bodytemp, "r", encoding='utf-8') as f:
                html= f.read()

            email_from = '1911047@nec.edu.in'
            epassword = 'jnfulpwymmsjvpqo'
            email_to = EMAIL

            date_str = pd.Timestamp.today().strftime('%Y-%m-%d')
            
            email_message = MIMEMultipart()
            email_message['From'] = email_from
            email_message['To'] = email_to
            email_message['Subject'] = f'Report email - {date_str}'

            email_message.attach(MIMEText(html, "html"))
            
            email_string = email_message.as_string()

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(email_from, epassword)
                server.sendmail(email_from, email_to, email_string)
            return render_template('notify.html')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
    return render_template('ibmprojectlogin.html',imgg3=img3)


@app.route('/login')
def login():
    if request.method == 'POST' :
        NAME = request.form['NAME']
        PASSWORD = request.form['PASSWORD']
        sql = "SELECT * FROM users WHERE NAME =? AND PASSWORD=?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt,1,NAME)
        ibm_db.bind_param(stmt,2,PASSWORD)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        print (account)
        
        if account:
            session['loggedin'] = True
            session['id'] = account['USERNAME']
            userid=  account['USERNAME']
            session['username'] = account['USERNAME']
            msg = 'Logged in successfully !'
            
            return redirect(url_for('predict'))
        else:
            msg = 'Incorrect username / password !'
            

    img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
    return render_template('ibmprojectlogin.html',imgg3=img3)




@app.route('/predict')
def predict():
    img4 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmpredict.jpg')
    return render_template('ibmprojectprediction.html',imgg4=img4)

@app.route('/result')
def result():
    if request.method == "POST":
        f = request.files['input_img']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(str(basepath), 'User_Images', str(f.filename))
        f.save(filepath)
        img = image.load_img(filepath, target_size=(299, 299))
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0)  
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['No_DR','Mild','Moderate','Proliferate_DR','severe']
        result = str(index[prediction[0]])
        print(result)
        return render_template('ibmresult.html')

@app.route('/confirm')
def confirmation():
    insert_sql = "INSERT INTO Users (NAME, EMAIL, PASSWORD)  VALUES (?,?,?)"
    prep_stmt = ibm_db.prepare(conn, insert_sql)
    ibm_db.bind_param(prep_stmt, 1, var_list[0])
    ibm_db.bind_param(prep_stmt, 2, var_list[1])
    ibm_db.bind_param(prep_stmt, 3, var_list[2])
    ibm_db.execute(prep_stmt)
    Img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
    return render_template('ibmprojectlogin.html',Imgg3=Img3)
 

@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   Img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
   return render_template('ibmprojectlogin.html',Imgg3=Img3)


# main driver function
if __name__ == '__main__':

	app.run()
