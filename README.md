Inventory Management System : 
This project is a web-based inventory management system that uses Flask for the web application framework, Pandas for data manipulation, and ARIMA for time series forecasting. It allows for the management of stock levels, demand forecasting, and inventory insights to aid in decision-making for optimal stock control. The application includes an integration with Power BI for reporting.


Features
1.	Inventory Management: Add, update, and subtract product stock quantities.
2.	Demand Forecasting: Predicts future demand using ARIMA models.
3.	Inventory Insights: Calculates optimal stock levels, safety stock, and Economic Order Quantity (EOQ) for each product.
4.	Visualization: View demand forecasts vs. actuals through plotted graphs.
5.	ETL Explanation: Explanation page for the data extraction, transformation, and loading process.
6.	Power BI Integration: Embedded Power BI report for detailed inventory analytics.


 Requirements
1.	Python 3.6+
2.	Flask- for building the web application.
3.	Pandas - for handling data and Excel files.
4.	Statsmodels- for ARIMA forecasting.
5.	Scikit-Learn - for calculating Mean Squared Error.
6.	Matplotlib- for plotting forecast graphs.


Installation
1.	Clone the repository:  
bash   
git clone  https://github.com/HalchalGothwad/inventorymanagementproject
cd inventorymanagementproject
2.	Install dependencies: 
pip install -r requirements.txt
3.	Set up data files:
 Ensure that stock.xlsx and pastorders.xlsx are available in the data directory as specified in the code.


Configuration
Power BI URL: Update the power_bi_url in the code with your Power BI report's embed link.  

Usage
Run the application:
bash
Copy code
python app.py
Open your browser and go to http://127.0.0.1:5000/.
Use the interface to add, update, and subtract stock quantities, view forecast data, and explore inventory insights.


Key Functions
ARIMA Forecasting: Generates a 4-month forecast for each product based on past order data.
Inventory Insights Calculation: Uses safety stock and EOQ calculations to recommend stock levels and reorder points.
Plot Forecast: Plots forecast vs. actual data for visual analysis.
API Endpoints
1. /: Homepage showing inventory and forecast results.
2. /add_product: Adds a new product to inventory.
3. /update_stock: Updates stock for an existing product.
4. /subtract_stock: Reduces stock for an existing product.
5. /view_forecast: Displays forecasted demand vs. actual data.
6. /inventory_insights: Shows inventory insights for reorder points and stockout risks.
7. /etl_explanation: Explanation of the ETL process.


File Structure
(i) app.py: Main application file.
(ii) data/: Contains stock.xlsx and pastorders.xlsx files.
(iii) static/: Stores generated forecast plot images.
(iv) templates/: HTML templates for rendering views.


Acknowledgements
This project utilizes open-source libraries such as Pandas, Statsmodels, Scikit-Learn, and Matplotlib. Special thanks to the Power BI team for providing data visualization support
