import requests
import sys

def get_user_input(prompt, type_cast=float, validation_fn=None):
    while True:
        try:
            value = type_cast(input(prompt))
            if validation_fn and not validation_fn(value):
                raise ValueError("Input did not meet validation criteria.")
            if value < 0:
                raise ValueError("Value cannot be below 0.")
            return value
        except ValueError:
            print("Invalid input. Please try again.")

def predict_demand():
    print("\n=== Regional Reference Price (RRP) Prediction Interface ===\n")
    
    # Collect inputs
    data = {
        "demand": get_user_input("Enter demand value: "),
        "demand_pos_RRP": get_user_input("Enter positive RRP demand: "),
        "demand_neg_RRP": get_user_input("Enter negative RRP demand: "),
        "min_temperature": get_user_input("Enter minimum temperature: "),
        "max_temperature": get_user_input("Enter maximum temperature: "),
        "solar_exposure": get_user_input("Enter solar exposure: "),
        "rainfall": get_user_input("Enter rainfall: "),
        "frac_at_neg_RRP": get_user_input("Enter fraction at negative RRP (0.0 - 1.0): ", float, lambda x: 0.0 <= x <= 1.0),
        "month": get_user_input("Enter month (1-12): ", int, lambda x: 1 <= x <= 12),
        "school_day": get_user_input("Enter school day (0 or 1): ", int, lambda x: x in [0, 1])
    }

    try:
        # Make API call
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "Predicted RRP" in result:
                prediction = result["Predicted RRP"].replace("$", "")
                print(f"\nPredicted RRP: ${float(prediction):.2f}")
            else:
                print("\nError: 'Predicted RRP' key not found in the response.")
        else:
            print(f"\nError: API returned status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API. Is the server running?")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    while True:
        predict_demand()
        if input("\nWould you like to make another prediction? (y/n): ").lower() != 'y':
            break
