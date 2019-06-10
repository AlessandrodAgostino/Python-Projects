import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template

#Function that returns name and emails from a file rubrica
def get_contacts(filename):
    """Read the name and email address from a file in the format:

        alessandro alessandro.dagostino96@gmail.com
        riccardo riccardo.scheda@gmail.com
    """
    names = []
    emails = []
    with open(filename, mode='r', encoding='utf-8') as contacts_file:
        for a_contact in contacts_file:
            names.append(a_contact.split()[0])
            emails.append(a_contact.split()[1])
    return names, emails

#Function that returns the content of a file as a Template
def read_template(filename):
    """Template words have to be written as:

        Dear ${PERSON_NAME}, ...
    """
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)


def main():
    #User-ID and password for the connection on gmail
    username = 'alessandro.dagostino.notifica@gmail.com'
    password = 'notific@'

    #Not full-safe connection to the mail server
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(username,password)

    #Exctract the names, emails and text
    names, emails = get_contacts('/home/alessandro/Python/email/my_contacts.txt')
    message_template = read_template('/home/alessandro/Python/email/message.txt')

    for name, email in zip(names, emails):
        msg = MIMEMultipart()
        msg['From']=username
        msg['To']= email
        msg['Subject']='Messaggio inviato automaticamente'
        message = message_template.substitute(PERSON_NAME=name.title())

        msg.attach(MIMEText(message, 'plain'))
        server.send_message(msg)
        del msg

if __name__ == '__main__':
    main()
