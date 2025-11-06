import os
import sys
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Just render the page with the input form
        return render_template('home.html')
    else:

        try:
            # Get the number of days from the form
            periods_to_forecast = int(request.form.get('periods'))
            
            # Validate input - UPDATED TO 180 DAYS (6 MONTHS)
            if periods_to_forecast < 1 or periods_to_forecast > 180:
                return render_template(
                    'home.html', 
                    error="Please enter a value between 1 and 180 days (6 months maximum)."
                )
            
            # Create CustomData object
            data = CustomData(periods=periods_to_forecast)

            # Get the data as a dataframe
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Prediction requested for {periods_to_forecast} periods.")

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()
            
            # Get the forecast dataframe (full forecast with historical)
            forecast_df = predict_pipeline.predict(pred_df)
            
            # Get the Prophet model for plotting
            model = predict_pipeline.get_model()
            
            logging.info("Prediction successful.")

            # Separate historical and forecast data
            forecast_data = forecast_df.tail(periods_to_forecast).copy()
            
            # Generate Prophet plots          
            # Plot 1: Main forecast plot with historical data
            fig1 = model.plot(forecast_df, figsize=(14, 6))
            ax1 = fig1.gca()
            ax1.set_title(f'{periods_to_forecast}-Day Sales Forecast', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Quantity Sold', fontsize=12)
            
            # Add vertical line at forecast start
            last_historical_date = forecast_df.iloc[-periods_to_forecast - 1]['ds']
            ax1.axvline(x=last_historical_date, color='red', linestyle='--', linewidth=2, label='Forecast Start')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            prophet_forecast_img = fig_to_base64(fig1)
            
            # Plot 2: Components plot (trend, weekly, yearly seasonality)
            fig2 = model.plot_components(forecast_df, figsize=(14, 10))
            prophet_components_img = fig_to_base64(fig2)
            
            # Calculate summary statistics
            forecast_summary = {
                'total_predicted': int(forecast_data['yhat'].sum()),
                'average_daily': int(forecast_data['yhat'].mean()),
                'min_daily': int(forecast_data['yhat'].min()),
                'max_daily': int(forecast_data['yhat'].max()),
                'peak_date': forecast_data.loc[forecast_data['yhat'].idxmax(), 'ds'].strftime('%Y-%m-%d'),
                'low_date': forecast_data.loc[forecast_data['yhat'].idxmin(), 'ds'].strftime('%Y-%m-%d'),
                'std_daily': int(forecast_data['yhat'].std()),
                'confidence_avg': int((forecast_data['yhat_upper'] - forecast_data['yhat_lower']).mean() / 2)
            }
            
            # Format table data (keep for potential future use, but won't display)
            forecast_table = forecast_data.copy()
            forecast_table['ds'] = forecast_table['ds'].dt.strftime('%Y-%m-%d')
            forecast_table['yhat'] = forecast_table['yhat'].round(0).astype(int)
            forecast_table['yhat_lower'] = forecast_table['yhat_lower'].round(0).astype(int)
            forecast_table['yhat_upper'] = forecast_table['yhat_upper'].round(0).astype(int)
            
            results_list = forecast_table.to_dict(orient='records')

            # Render template with all data
            return render_template(
                'home.html', 
                results=results_list, 
                periods=periods_to_forecast,
                prophet_forecast_img=prophet_forecast_img,
                prophet_components_img=prophet_components_img,
                summary=forecast_summary
            )
        
        except Exception as e:
            logging.error(f"Error in prediction: {e}", exc_info=True)
            return render_template(
                'home.html',
                error=f"An error occurred: {str(e)}"
            )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)