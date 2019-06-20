#Sending emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(message):
    username = 'alessandro.dagostino.notifica@gmail.com'
    password = 'notific@'
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(username,password)

    msg = MIMEMultipart()
    msg['From']=username
    msg['To']= 'alessandro.dagostino96@gmail.com'
    msg['Subject']='prova func'

    msg.attach(MIMEText(message, 'plain'))
    server.send_message(msg)
    del msg

send_email("\nprova funzione{}".format(2))
