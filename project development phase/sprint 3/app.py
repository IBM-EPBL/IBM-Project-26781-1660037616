import smtplib, ssl
## email.mime subclasses
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
## The pandas library is only for generating the current date, which is not necessary for sending emails
import pandas as pd
import os
from datetime import datetime

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
        print(fdfghg)
        if account:
            session['loggedin'] = True
            session['id'] = account['USERNAME']
            userid=  account['USERNAME']
            session['username'] = account['USERNAME']
            msg = 'Logged in successfully !'
            print("sss")
            # return render_template('register.html', regImg=loginImg)
            return redirect(url_for('predict'))
        else:
            msg = 'Incorrect username / password !'
            print(nj)
    #return render_template('login.html', msg = msg,loginImg=loginImg)

    img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
    return render_template('ibmprojectlogin.html',imgg3=img3)


@app.route('/register')
def register():
    # img = os.path.join(app.config['UPLOAD_FOLDER'],'diabetic-retinopathy-illustration_42265-59.webp')
    # img1 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.jpg')
    # img2 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmlogin.jpg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmregister.png')
    return render_template('ibmprojectregister.html',imgg3=img3)

@app.route('/predict')
def predict():
    img4 = os.path.join(app.config['UPLOAD_FOLDER'],'ibmpredict.jpg')
    return render_template('ibmprojectprediction.html',imgg4=img4)

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


            # Set up the email addresses and password. Please replace below with your email address and password
            email_from = '1911047@nec.edu.in'
            epassword = 'jnfulpwymmsjvpqo'
            email_to = EMAIL

            # Generate today's date to be included in the email Subject
            date_str = pd.Timestamp.today().strftime('%Y-%m-%d')

            # Create a MIMEMultipart class, and set up the From, To, Subject fields
            email_message = MIMEMultipart()
            email_message['From'] = email_from
            email_message['To'] = email_to
            email_message['Subject'] = f'Report email - {date_str}'

            # Attach the html doc defined earlier, as a MIMEText html content type to the MIME message
            email_message.attach(MIMEText(html, "html"))
            # Convert it as a string
            email_string = email_message.as_string()

            # Connect to the Gmail SMTP server and Send Email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(email_from, epassword)
                server.sendmail(email_from, email_to, email_string)
            return render_template('notify.html')
            
    return render_template('ibmprojectlogin.html')


@app.route('/confirm')
def confirmation():
    insert_sql = "INSERT INTO Users (NAME, EMAIL, PASSWORD)  VALUES (?,?,?)"
    prep_stmt = ibm_db.prepare(conn, insert_sql)
    ibm_db.bind_param(prep_stmt, 1, var_list[0])
    ibm_db.bind_param(prep_stmt, 2, var_list[1])
    ibm_db.bind_param(prep_stmt, 3, var_list[2])
    ibm_db.execute(prep_stmt)
    return render_template('ibmprojectlogin.html')
 
@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   regImg = os.path.join(app.config['UPLOAD_FOLDER'],'Plasma_login.png')
   return render_template('login.html',regImg=regImg)


# main driver function
if __name__ == '__main__':
	# run() method of Flask class runs the application
	# on the local development server.
	app.run()
