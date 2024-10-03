from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

try:
    # Try relative import when used as part of a package
    from .db_operations import *
except ImportError:
    # Fallback to absolute import when running directly
    from db_operations import *


# @tool 
# def search_user_info() -> str:
#     "Run this tool to retrieve the customer information"
#     return """Customer Info
#     Jon Doe
#     Number +1 12345
#     email: jonhdoe@email.com
#     customerId: 0 
#     CustomerScore: 700

#     Current products:
#                 - Mortgage: 7 Year - 10.2%
#                     Amount: 200,000$ """

@tool 
def search_user_info(config: RunnableConfig) -> list[dict]:
    """Run this tool to retrieve the customer information
      Returns:
        A list of dictionaries where each dictionary contains the customer information details """
    
    configuration = config.get("configurable", {})
    customer_id = configuration.get("customer_id", None)
    if not customer_id:
        raise ValueError("No customer ID configured.")

    info = get_customer_info(customer_id)
    return info

@tool
def update_customer_info(config:RunnableConfig, new_address) -> str: 
    "Run this tool to update the record of customer address in the database"

    configuration = config.get("configurable", {})
    customer_id = configuration.get("customer_id", None)
    if not customer_id:
        raise ValueError("No customer ID configured")
    
    res = update_customer_address(new_address, customer_id)
    return res

@tool 
def send_mfa_code(config:RunnableConfig)-> str: 
    """ Run this tool when needed to send a multifactor authentication to the customer 
    """
    
    configuration = config.get("configurable", {})
    customer_id = configuration.get("customer_id", None)
    print(customer_id)
    if not customer_id:
        raise ValueError("No customer ID configured")
    info = get_customer_info(customer_id)
    number = info['customer_phone']
    print(f"number: {number}")

    return f"Code sent to number {number}"
