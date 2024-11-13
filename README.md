# Energy Price Forecasting: Predicting Regional Reference Price in Victoria

## Description

This project aims to predict the Regional Reference Price (RRP) in Victoria, Australia, using a dataset covering 2016 days from 1 January 2015 to 6 October 2020. The RRP reflects the recommended retail price for electricity in AUD$ per MWh. Unique to Victoria's energy market, some intervals exhibit a negative RRP, where energy producers effectively pay consumers, highlighting the importance of accurate forecasting. Below is a description of the [dataset](./data/energy_complete.csv)
 fields:

- **date** : datetime, the date of the recording
- **demand** : float, the total daily electricity demand in MWh
- **RRP** : float, the recommended retail price in AUD$ per MWh
- **demand_pos_RRP** : float, the total daily demand when RRP was positive, in MWh
- **RRP_positive** : float, the average positive RRP, weighted by the corresponding intraday demand, in AUD$ per MWh
- **demand_neg_RRP** : float, the total daily demand when RRP was negative, in MWh
- **RRP_negative** : float, the average negative RRP, weighted by the corresponding intraday demand, in AUD$ per MWh
- **frac_at_neg_RRP** : float, the fraction of the day when demand was traded at a negative RRP
- **min_temperature** : float, the minimum temperature during the day in Celsius
- **max_temperature** : float, the maximum temperature during the day in Celsius
- **solar_exposure** : float, the total daily sunlight energy in MJ/m²
- **rainfall** : float, the daily rainfall in mm
- **school_day** : boolean, indicates if students were in school on that day
- **holiday** : boolean, indicates if the day was a state or national holiday

This dataset provides key variables for training a machine learning model to predict RRP, supporting strategic planning for energy producers and aiding in market stability.

## Technologies Used
- **FastAPI**: for building the web service API
- **Uvicorn**: for running the ASGI server in FastAPI applications
- **Pandas**: for data manipulation and analysis
- **Matplotlib**: for data visualization
- **Seaborn**: for statistical data visualization
- **Scikit-Learn**: for building and training the machine learning model
- **Requests**: for making HTTP requests
- **Pipenv**: for environment and dependency management
- **Docker**: for containerization and deployment
- **IPykernel**: for Jupyter notebook support in the development environment

## Project Setup

#### This builds a docker application which trains the model and then starts up a web service (FastAPI).

```sh
pipenv install
docker build -t electricity_price . --no-cache
docker run -p 8000:8000 electricity_price
```

#### Note: Run the commands in the root directory of the project. It may take some time to see an output so dont think its stuck, the output is shown only after the final model has been trained and saved to a file. Also make sure port 8000 is not in use or the above commands may fail.


## Web service test (FastAPI)

### Test API call using curl

#### If you want to quickly test if the web service is working or not you can run the following command in your command line. 

```sh
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "demand": 800,
    "demand_pos_RRP": 300,
    "demand_neg_RRP": 150,
    "min_temperature": 12,
    "max_temperature": 25,
    "solar_exposure": 10,
    "rainfall": 0.5,
    "frac_at_neg_RRP": 0.2,
    "month": 7,
    "school_day": 1
  }'
```
#### Note: Make sure the web service is running first.

### Test API call using command line interface

#### If you want a more intuitive method of testing the web service you can run the following script. It will give you a command line interface so you can test the web service.

```sh
pipenv run python test.py
```

#### A successful run should look like this.

```sh
=== Regional Reference Price (RRP) Prediction Interface ===

Enter demand value: 800
Enter positive RRP demand: 300
Enter negative RRP demand: 150
Enter minimum temperature: 12
Enter maximum temperature: 25
Enter solar exposure: 10
Enter rainfall: 0.5
Enter fraction at negative RRP (0.0 - 1.0): 0.2
Enter month (1-12): 7
Enter school day (0 or 1): 1

Predicted RRP: $57.87

Would you like to make another prediction? (y/n): n
```

#### Note: Make sure the web service is running first and run the command from root project directory.

## Background

- **FastAPI**: FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. It is designed to be easy to use and highly performant, leveraging asynchronous programming and type hints to provide faster response times and easier code maintenance. FastAPI is widely used for building RESTful APIs due to its automatic documentation generation and support for both synchronous and asynchronous request handling.

- **Uvicorn**: Uvicorn is a lightning-fast ASGI server implementation, used to serve ASGI-compatible frameworks like FastAPI. It is built on top of the `uvloop` event loop, which enhances speed and concurrency in handling web requests. Uvicorn’s performance and lightweight nature make it an ideal choice for running high-performance, asynchronous web applications in production.
