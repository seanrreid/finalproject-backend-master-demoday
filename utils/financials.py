from utils.data_fetching import (
    get_population_data,
    get_average_spending,
    get_industry_interest_rate
)
import pandas as pd

def get_average_spending(industry):
    # Placeholder values; replace with actual data fetching
    industry_spending = {
        'Fitness Center': 50,  # Monthly spending per customer
        'Coffee Shop': 35,     # Average $35/month per customer
        'Restaurant': 45,
        'Retail Store': 40,
        'Beauty Salon': 60
    }
    return industry_spending.get(industry, 40)

def get_industry_interest_rate(industry):
    # Placeholder values
    interest_rates = {
        'Fitness Center': 0.05,  # 5% of population
        'Coffee Shop': 0.15,     # 15% of population
        'Restaurant': 0.10,
        'Retail Store': 0.08,
        'Beauty Salon': 0.06
    }
    return interest_rates.get(industry, 0.3)  # Default value

def determine_market_share(competitors):
    total_competitors = len(competitors) + 1  # Including the new business
    market_share = 1 / total_competitors
    return market_share

def estimate_annual_revenue(business_idea, location, competitors):
    population = get_population_data(location) or 50000  # Default value if None
    interest_rate = get_industry_interest_rate(business_idea)
    market_size = population * interest_rate
    avg_spending = get_average_spending(business_idea)
    market_share = determine_market_share(competitors)
    annual_revenue = market_size * avg_spending * market_share
    return annual_revenue

def estimate_startup_costs(business_idea):
    startup_costs = {
        'Fitness Center': 250000,
        'Coffee Shop': 80000,
        'Restaurant': 120000,
        'Retail Store': 100000,
        'Beauty Salon': 60000
    }
    return startup_costs.get(business_idea, 75000)

def estimate_operational_expenses(business_idea, location):
    expenses = {
        'Fitness Center': 200000,
        'Coffee Shop': 80000,
        # Add more industries
    }
    return expenses.get(business_idea, 50000)  # Default value

def generate_p_and_l_statement(revenue, cost_of_goods_sold, operational_expenses):
    gross_profit = revenue - cost_of_goods_sold
    net_profit = gross_profit - operational_expenses
    return {
        'revenue': revenue,
        'cost_of_goods_sold': cost_of_goods_sold,
        'gross_profit': gross_profit,
        'operational_expenses': operational_expenses,
        'net_profit': net_profit
    }
