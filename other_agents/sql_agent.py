from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()


basic_banking_needs(customer_df.iloc[0], customer_products_df, products_df) 
filter_by_lifecycle_stage(customer_df.iloc[0], customer_products_df) 
basic_investment_and_savings(customer_df.iloc[0], customer_products_df, products_df) 
retirement_and_education_savings(customer_df.iloc[0], customer_products_df, products_df) 
loans_and_credit(customer_df.iloc[0], customer_products_df, products_df) 
insurance(customer_df.iloc[0], customer_products_df, products_df)