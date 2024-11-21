import openai
from utils.config import settings
import logging
from openai import OpenAIError

def generate_action_plan(evaluation_result):
    openai.api_key = settings.openai_api_key
    
    # Handle optional fields safely
    risks_text = ', '.join([risk.risk for risk in (evaluation_result.risks or [])])
    competitors_count = len(evaluation_result.competitors or [])
    mitigation_strategies = ', '.join(evaluation_result.mitigation_strategies or [])
    
    # Handle financial projections safely
    financial_info = ""
    if evaluation_result.financial_projection:
        financial_info = (
            f"Revenue - {evaluation_result.financial_projection.revenue}, "
            f"Net Profit - {evaluation_result.financial_projection.net_profit}"
        )
    
    prompt = (
        f"You're an experienced business consultant. Based on the following evaluation results, "
        f"provide a detailed action plan for the user:\n\n"
        f"Business Idea: {evaluation_result.business_idea}\n"
        f"Location: {evaluation_result.location}\n"
        f"Economic Indicator: {evaluation_result.economic_indicator}\n"
        f"Competitors: {competitors_count}\n"
        f"Risks: {risks_text}\n"
        f"Mitigation Strategies: {mitigation_strategies}\n"
        f"Financial Projections: {financial_info}\n\n"
        f"Provide a step-by-step action plan to help the user successfully start and grow their business."
    )
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            n=1,
        )
        action_plan = response.choices[0].text.strip()
        return action_plan
    except OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return "Unable to generate action plan at this time."
