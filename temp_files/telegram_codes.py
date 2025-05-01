import requests
import json




bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
def get_chat_id():
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        # Use json.dumps to print the JSON data in a formatted way
        print(json.dumps(data, indent=4))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

get_chat_id()






# bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
# chat_id = '-1002695103879'
# message = 'ALERT: Intrusion Detected'

# def send_telegram_message(bot_token, chat_id, message):
#     url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
#     payload = {
#         "chat_id": chat_id,
#         "text": message
#     }
#     requests.post(url, data=payload)


# send_telegram_message(bot_token, chat_id, message)





# bot_token = "8028477815:AAFjJLBcIltk3xgBlwmGRrLhgdH378WNOH4"
# chat_id = '-1002695103879'

# def send_photo(photo_path, caption=None):
#     url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
#     with open(photo_path, "rb") as photo:
#         payload = {
#             "chat_id": chat_id,
#             "caption": caption or ""
#         }
#         files = {
#             "photo": photo
#         }
#         response = requests.post(url, data=payload, files=files)
#         print(response.json())

# # Example usage
# send_photo("camera_alert.jpg", caption="🚨 Motion detected!")
