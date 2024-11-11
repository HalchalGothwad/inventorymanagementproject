import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, render_template, request, redirect, url_for, flash
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)

# Paths to data files
excel_file_path = r'C:\Users\Lenovo\Documents\inventory\inventorymanagementproject\data\stock.xlsx'
power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=522f1632-86c9-4dd1-a2c4-5e7055a898de&autoAuth=true&ctid=e5ba4765-79e6-4ae2-8686-668b280f722c"
past_orders_file_path = r'C:\Users\Lenovo\Documents\inventory\inventorymanagementproject\data\pastorders.xlsx'

# Load past orders data
try:
    past_orders_data = pd.read_excel(past_orders_file_path)
    # Clean column names by stripping spaces
    past_orders_data.columns = past_orders_data.columns.str.strip()

    # Debugging: Print column names and sample of the data
    print("Columns in past orders file:", past_orders_data.columns)
    print("First few rows of past orders data:")
    print(past_orders_data.head())

    # Ensure 'PDName' is in the columns
    if 'PDName' not in past_orders_data.columns or 'Order Date' not in past_orders_data.columns or 'Order Quantity' not in past_orders_data.columns:
        raise KeyError("'PDName', 'Order Date', or 'Order Quantity' column not found in pastorders Excel file.")

    # Convert 'Order Date' to datetime and handle invalid dates
    past_orders_data['Order Date'] = pd.to_datetime(past_orders_data['Order Date'], errors='coerce')
    past_orders_data = past_orders_data.dropna(subset=['Order Date'])  # Drop rows with invalid dates

    print("Cleaned past orders data:")
    print(past_orders_data.head())
except Exception as e:
    print(f"Error loading past orders data: {e}")
    flash(f"Error loading past orders data: {e}", 'danger')

# Load SKU (inventory) data
try:
    sku_data = pd.read_excel(excel_file_path)
    # Clean column names by stripping spaces
    sku_data.columns = sku_data.columns.str.strip()

    # Debugging: Print column names and sample of the data
    print("Columns in SKU data file:", sku_data.columns)
    print("First few rows of SKU data:")
    print(sku_data.head())

    # Ensure 'PDName' and 'Units' are in the columns
    if 'PDName' not in sku_data.columns or 'Units' not in sku_data.columns:
        raise KeyError("'PDName' or 'Units' column not found in the Excel file.")

    # Set 'PDName' as the index and remove duplicates
    sku_data = sku_data.drop_duplicates(subset='PDName').set_index('PDName')
    inventory = sku_data.to_dict('index')  # Convert to dictionary for easy access

    print("Cleaned SKU data (inventory):")
    print(sku_data.head())
except Exception as e:
    print(f"Error loading SKU data: {e}")
    flash(f"Error loading SKU data: {e}", 'danger')

# Initialize forecast_results for products
forecast_results = {}
for pdname, group in past_orders_data.groupby('PDName'):
    ts_data = group.set_index('Order Date')['Order Quantity'].resample('M').sum().fillna(0)

    try:
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=4)

        if len(ts_data) >= 4:
            train_data, test_data = ts_data[:-4], ts_data[-4:]
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            test_forecast = fitted_model.forecast(steps=4)
            mse = mean_squared_error(test_data, test_forecast)
            mbe = np.mean(test_forecast - test_data)

            forecast_results[pdname] = {
                'actual': ts_data,
                'forecast': forecast,
                'mse': mse,
                'mbe': mbe
            }
        else:
            forecast_results[pdname] = {
                'actual': ts_data,
                'forecast': forecast,
                'mse': None,
                'mbe': None
            }
    except Exception as e:
        flash(f"Error forecasting for {pdname}: {e}", 'danger')

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html', inventory=inventory, power_bi_url=power_bi_url, forecast_results=forecast_results)

@app.route('/add_product', methods=['POST'])
def add_product():
    pdname = request.form['pdname']
    current_stock_quantity = request.form.get('current_stock_quantity', type=int)
    units = request.form['units']
    unit_price = request.form.get('unit_price', type=float)

    if pdname in inventory:
        flash(f"Product {pdname} already exists.", 'danger')
        return redirect(url_for('home'))

    new_product = {'Current Stock Quantity': current_stock_quantity, 'Units': units, 'Unit Price': unit_price}
    inventory[pdname] = new_product
    sku_data.loc[pdname] = new_product

    forecast_results[pdname] = {
        "actual": pd.Series(dtype=float),
        "forecast": pd.Series(dtype=float)
    }

    flash(f"Added new product {pdname} to the inventory.", 'success')
    sku_data.to_excel(excel_file_path, index=True)
    return redirect(url_for('home'))

@app.route('/update_stock', methods=['POST'])
def update_stock():
    pdname = request.form['pdname']
    quantity = request.form.get('quantity', type=int)

    if pdname not in inventory:
        flash(f'Product {pdname} not found', 'danger')
        return redirect(url_for('home'))

    inventory[pdname]['Current Stock Quantity'] += quantity
    sku_data.at[pdname, 'Current Stock Quantity'] = inventory[pdname]['Current Stock Quantity']
    flash(f'Updated stock for {pdname}: New quantity is {inventory[pdname]["Current Stock Quantity"]}', 'success')
    sku_data.to_excel(excel_file_path, index=True)
    return redirect(url_for('home'))

@app.route('/subtract_stock', methods=['POST'])
def subtract_stock():
    pdname = request.form['pdname']
    quantity = request.form.get('quantity', type=int)

    if pdname not in inventory:
        flash(f'Product {pdname} not found', 'danger')
        return redirect(url_for('home'))

    if inventory[pdname]['Current Stock Quantity'] < quantity:
        flash(f'Not enough stock to subtract.', 'danger')
        return redirect(url_for('home'))

    inventory[pdname]['Current Stock Quantity'] -= quantity
    sku_data.at[pdname, 'Current Stock Quantity'] = inventory[pdname]['Current Stock Quantity']
    flash(f'Updated stock for {pdname}: New quantity is {inventory[pdname]["Current Stock Quantity"]}', 'success')
    sku_data.to_excel(excel_file_path, index=True)
    return redirect(url_for('home'))

def calculate_safety_stock(demand_forecast, lead_time, service_level):
    demand_std_dev = demand_forecast.std()
    z_score = np.percentile([service_level], service_level * 100)
    safety_stock = z_score * demand_std_dev * np.sqrt(lead_time)
    return safety_stock

def calculate_eoq(annual_demand, order_cost, holding_cost):
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
    return eoq

service_level = 0.95
average_lead_time = 2
order_cost = 50
holding_cost_per_unit = 1.5

for pdname, forecast_data in forecast_results.items():
    demand_forecast = forecast_data['forecast']
    current_stock = inventory[pdname]['Current Stock Quantity']

    safety_stock = calculate_safety_stock(demand_forecast, average_lead_time, service_level)
    annual_demand = demand_forecast.sum() * 12
    eoq = calculate_eoq(annual_demand, order_cost, holding_cost_per_unit * 52)
    optimal_stock_level = eoq + safety_stock

    forecast_results[pdname]['optimal_stock_level'] = optimal_stock_level
    forecast_results[pdname]['safety_stock'] = safety_stock
    forecast_results[pdname]['eoq'] = eoq

@app.route('/view_forecast', methods=['POST'])
def view_forecast():
    pdname = request.form['pdname']
    if pdname not in forecast_results or forecast_results[pdname]["actual"].empty:
        flash(f"Forecast data for '{pdname}' is either unavailable or lacks sufficient historical data.", 'danger')
        return redirect(url_for('home'))

    forecast_data = forecast_results[pdname]

    # Retrieve units from the inventory
    units = inventory.get(pdname, {}).get('Units', 'N/A')

    plt.figure(figsize=(10, 4))
    plt.plot(forecast_data['actual'].index, forecast_data['actual'], label='Actual')
    plt.plot(forecast_data['forecast'], label='Forecast')
    plt.title(f'Forecast vs Actual for {pdname}')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.grid(True)
    forecast_image_path = os.path.join('static', f'{pdname}_forecast.png')
    plt.savefig(forecast_image_path)
    plt.close()

    return render_template('forecast_view.html', pdname=pdname, forecast_data=forecast_data, units=units, forecast_image=forecast_image_path)

@app.route('/inventory_insights')
def inventory_insights():
    lead_time_weeks = 2  # Example: 2 weeks lead time
    service_level = 0.95  # 95% service level
    insights = calculate_inventory_insights(inventory, forecast_results, lead_time_weeks, service_level)
    return render_template('inventory_insights.html', insights=insights)

def calculate_inventory_insights(inventory, forecast_results, lead_time_weeks, service_level):
    insights = {}
    for pdname, data in forecast_results.items():
        forecasted_demand = data['forecast']
        current_stock = inventory.get(pdname, {}).get('Current Stock Quantity', 0)
        units = inventory.get(pdname, {}).get('Units', 'N/A')

        # Calculate safety stock
        safety_stock = calculate_safety_stock(forecasted_demand, lead_time_weeks, service_level)

        # Calculate the average weekly demand
        avg_weekly_demand = forecasted_demand.mean()

        # Calculate reorder point
        reorder_point = avg_weekly_demand * lead_time_weeks + safety_stock

        # Check if the product needs to be reordered
        needs_reorder = current_stock <= reorder_point

        # Check for potential stockout (if current stock is less than the forecasted cumulative demand)
        potential_stockout = current_stock < forecasted_demand.cumsum().iloc[0]

        insights[pdname] = {
            'current_stock': current_stock,
            'units': units,
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'needs_reorder': needs_reorder,
            'potential_stockout': potential_stockout
        }

    return insights
@app.route('/etl_explanation')
def etl_explanation():
    return render_template('etl_explanation.html')

if __name__ == '__main__':
    app.run(debug=True)




