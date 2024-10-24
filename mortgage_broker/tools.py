from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import json
import requests
try:
    # Try relative import when used as part of a package
    from policies import qa_chain
except ImportError:
    # Fallback to absolute import when running directly
    from .policies import qa_chain


# @tool 
# def search_user_info(config: RunnableConfig) -> list[dict]:
#     """Run this tool to retrieve the customer information
#       Returns:
#         A list of dictionaries where each dictionary contains the customer information details """
    
#     configuration = config.get("configurable", {})
#     customer_id = configuration.get("customer_id", None)
#     if not customer_id:
#         raise ValueError("No customer ID configured.")

#     info = get_customer_info(customer_id)
#     return info

# @tool
# def update_customer_info(config:RunnableConfig, new_address) -> str: 
#     "Run this tool to update the record of customer address in the database"

#     configuration = config.get("configurable", {})
#     customer_id = configuration.get("customer_id", None)
#     if not customer_id:
#         raise ValueError("No customer ID configured")
    
#     res = update_customer_address(new_address, customer_id)
#     return res

# @tool 
# def send_mfa_code(config:RunnableConfig)-> str: 
#     """ Run this tool when needed to send a multifactor authentication to the customer 
#     """
    
#     configuration = config.get("configurable", {})
#     customer_id = configuration.get("customer_id", None)
#     print(customer_id)
#     if not customer_id:
#         raise ValueError("No customer ID configured")
#     info = get_customer_info(customer_id)
#     number = info['customer_phone']
#     print(f"number: {number}")

#     return f"Code sent to number {number}"

@tool 
def create_ticket(title:str, subject:str) -> str:
    """Run this tool to create a ticket in the CRM for the current issue
    Args: 
        title: str title describing the type of the ticket
        subject: str A short description of the purpose and content of the ticket
    """
    return f'Ticket {title} succesfully created'

@tool
def send_documents_to_sign(email: str) -> str:
    """ Run this tools to send the documents for signature after the customer agrees to a term for his renewal.
     Args:
      email: str Customer email """
    return f"documents send to {email}"

@tool
def transfer_human_agent() -> str: 
    """Run this tool to transfer the call to a human agent"""
    return "Transfered to Human Agent"

@tool
def market_rates_tool():
    """ Run this tool to get the current available rates for the client, 
    dont present them directly to the client, but use it as a base for the renewal negotiation. 
    Always start offering from the higher bound, never offer a rate bellow the provided bound.
    
    """

    url = "https://api.browse.ai/v2/robots/4ce11b49-9b2c-44cf-9216-ac2d246b04a5/tasks/043d6a42-c716-4d6b-b811-17fa429b46a4"

    payload = json.dumps({})
    headers = {
    'Authorization': 'Bearer 445b59de-5c03-4aa2-877e-0d9a220c30f9:3d91accc-25ae-450a-8945-60eff61446ce',
    'Content-Type': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    j_response = json.loads(response.text)
    mortgage_lists = j_response['result']['capturedLists']['ratehub-rates']

    # result = []
    # for lender in mortgage_lists: 
    #     res = {key: lender[key] for key in lender.keys()
    #     & {'Lender Name', 'Interest Rate'}}
    #     result.append(res)

    return mortgage_lists

@tool
def inflation(years: int, price:int) -> int:
    """ Asses the value of the property based on the purchase price and the time that has passed from the purchase.
    Args: 
        years: Year of the purchase of the property
        price: Purchase price of the property. 
    """
    today = 2024
    return (today - years)*0.08*price + price

@tool
def property_assesment_tool(borrow_value:int, property_value:int) -> str: 
    """ Calculate the Loan-To-Value ratio after the customer provides the reamaining loan amount and property value
    Args: 
        borrow_value:int Amount borrowed in the loan. 
        property_value:int Stimated actual value of the properti
        """
    ltv = borrow_value/property_value * 100
    return "{:.2f}% LTV ratio".format(ltv)

@tool
def questionnaire_tool() -> str: 
    """ Run this tool the get the questionnaire to ask to the customer
    """
    return """Here's the list of questions: \n
1. When is your mortgage set to renew and who's your current mortgage provider?\n
2. How much is left on your mortgage balance, and what’s your estimate of the current value of your property?\n
3. Can you confirm which province your home is located in, and if your mortgage is insured by CMHC, Genworth Canada, or Canada Guarantee?
"""
            # 1. When is your mortgage set to renew?
            # 2. Who's your current mortgage provider?
        # 5. Which province is your home located in?
        #     6. Are you planning to stay in this home after the renewal?
        #     7. Is your mortgage insured by CMHC, Genworth Canada, or Canada Guarantee?

@tool
def save_questionnaire_tool(query):
    """ Run this tool to save the answers provided by the customer
    Args: 
    query: list[dict] ex. [{'question': 'question1', 'answer': 'answer1'}, ...] """

    return query


@tool
def knowledge_base_tool(query:str)-> str:
    """Use this tool to answer customer questions and retrieve information related to BLD and lender policies, mortgages and more. 
    Args:
    query: (str) the user question"""
    response = qa_chain.invoke(query)
    return response

if __name__ == "__main__":

    r = knowledge_base_tool.invoke('How does your renewal process work, especially for customers switching from another lender?"')
    print(r)