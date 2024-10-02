import requests

def send_pushover_message(title, message):
    url = "https://api.pushover.net/1/messages.json"

    data = {
        "token": "am2afn47rup1dtbik2s9jcz79eu3bx",
        "user": "uy2dbt84ncg8uathk2fn9pi17zbuz9",
        "device": "mi10tpro",
        "title": title,
        "message": message
    }

    response = requests.post(url, data=data)

    print(response.text)

send_pushover_message("Test", "This is a test message")