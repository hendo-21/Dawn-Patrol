"""
This module supports firing a push notification via Pushcut providing users
with a helpful notification about surfer count.
"""

import requests


def send_notification(surfer_count, api_key, endpoint_url):
    # Send a notification if there's an issue with capture.
    if surfer_count is None:
        requests.post(endpoint_url, json={"text": "Could not get surfer count this morning. Check stream for issues."}, headers={"API-Key": api_key})
        print("Error with frame capture. Check stream URL.")
        return

    msg = ""
    # Define three different messages
    if 0 <= surfer_count <= 20:
        msg = f"🏄‍♂️ It's empty! There are approximately {surfer_count} in the water. Go get it!"
    elif 21 <= surfer_count <= 50:
        msg = f"⚠️ You'd be out there with {surfer_count} of your closest friends, but you could still catch a few."
    else:
        msg = f"❌ Find something else to do today. It's packed: {surfer_count} surfers in the water."

    # Send the request depending on the message
    response = requests.post(endpoint_url, json={"text": msg}, headers={"API-Key": api_key})
    print(f"Surfer count notification posted successfully with status {response.status_code}")
