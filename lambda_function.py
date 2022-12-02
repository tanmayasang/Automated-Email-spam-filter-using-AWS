import json
import boto3
import logging
import os
import email
from email.policy import default as default_policy
import spam_classifier_utilities as utilities

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

SAGE_ENDPOINT = os.environ['SAGE_ENDPOINT']

def send_email(message, email):
    logger.info('sending email')
    ses = boto3.client('ses')
    ses_response = ses.send_email(
    Source='abirbhavdutta@gmail.com',
    Destination={
        'ToAddresses': [
            email,
        ],
        'CcAddresses': [
            'abirbhavdutta@gmail.com'
        ]
    },
    Message={
        'Subject': {
            'Data': 'About your email'
        },
        'Body': {
            'Text': {
                'Data': message
            }
        }
    }
    )
    logger.info(f'ses response is {ses_response}')
    
def build_message(EMAIL_RECEIVE_DATE, EMAIL_SUBJECT, EMAIL_BODY, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE):
    message = f"We received your email sent at {EMAIL_RECEIVE_DATE} with \
the subject '{EMAIL_SUBJECT}'. \
Here is a 240 character sample of the email body: \
'{EMAIL_BODY[:240]}' The email was categorized as {CLASSIFICATION} with a \
{CLASSIFICATION_CONFIDENCE_SCORE}% confidence."
    return message

def remove_new_lines(body):
    return body.replace('\n', ' ').replace('\r', '').strip()

def get_encoded_messages(body):
    vocabulary_length = 9013
    messages = [body]
    #test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    one_hot_test_messages = utilities.one_hot_encode(messages, vocabulary_length)
    encoded_test_messages = utilities.vectorize_sequences(one_hot_test_messages, vocabulary_length)
    return encoded_test_messages

def lambda_handler(event, context):
    logger.info(f'event is {event}')
    
    # Extract Message
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    bucket_key = event["Records"][0]["s3"]["object"]["key"]
    
    s3 = boto3.client("s3")
    s3_response = s3.get_object(Bucket = bucket_name, Key = bucket_key)
    
    logger.info(f'Got object from s3. Response is {s3_response}')
    
    email_raw = s3_response['Body'].read()
    email_detail = email.message_from_bytes(email_raw, policy=default_policy)
    
    sender = email_detail['From']
    email_date = email_detail['Date']
    email_subject = email_detail['Subject']
    
    body = email_detail.get_body(preferencelist=('plain')).get_content()
    
    logger.info(f'sender = {sender}, date = {email_date}, subject = {email_subject}, body = {body}')

    # Preprocess
    body = remove_new_lines(body)
    encoded_test_messages = get_encoded_messages(body)
    logger.info(f'Encoded test messages: {encoded_test_messages}')
    payload = json.dumps(encoded_test_messages.tolist())
    
    
    # # Query Sagemaker
    sagemeaker = boto3.client('runtime.sagemaker')
    sage_response = sagemeaker.invoke_endpoint(EndpointName=SAGE_ENDPOINT,ContentType='application/json',Body=payload)
    
    logger.info(f'Response from sagemaker: {sage_response}')
    
    #Extract result from sagemaker response
    sage_response_body = sage_response['Body'].read().decode('utf-8')
    logger.info(f'Sage response body = {sage_response_body}')
    sage_body_obj = json.loads(sage_response_body)
    predicted_value = sage_body_obj['predicted_label'][0][0]
    classification = 'Ham' if predicted_value == 0 else 'Spam'
    confidence = sage_body_obj['predicted_probability'][0][0] * 100
    # # Build message
    email_message = build_message(email_date, email_subject, body, classification, confidence)
    logger.info(f'Email message to send is: {email_message}')
    # Send email
    send_email(email_message, sender)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
