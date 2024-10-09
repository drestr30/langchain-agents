import os
import requests
from dotenv import load_dotenv

# load_dotenv()

API_URL = "https://kapti-recommendations.azurewebsites.net/api"
Auth = "Ln6o_6SJHmauzzQNeA3gIn8LAziKj-rHDrc8B23NyZeBAzFunkHWzA%3D%3D"
   
def get_customer_info(customer_id: str) -> str:
    try:
        # Prepare payload for the API
        payload = {
            'customer_id': customer_id,
        }
        
        # Make the API call
        response = requests.post(f"{API_URL}/get_customer_info?code={Auth}", json=payload)
        
        if response.status_code == 200:
            r = response.json()  # Assume API returns a list of dictionaries
            return r
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def update_customer_address(new_address, customer_id) -> str:
    msg = "success"
    try:
        # Prepare payload for the API
        payload = {
            "customer_id" : customer_id,
            "address": new_address
        }

        # Make the API call
        response = requests.post(f"{API_URL}/update_customer_info?code={Auth}", json=payload)

        if response.status_code == 200:
            print("Update successful")
        else:
            msg = f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        print(f"An error occurred: {e}")
        msg = str(e)
    return msg

if __name__ == "__main__":
    res = get_customer_info(1)
    print(res)

