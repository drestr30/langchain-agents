import json
import time
import azure.functions as func
import logging
from db_operations import db_get_customer_info, db_update_customer_address, connect_db


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="get_customer_info", auth_level=func.AuthLevel.FUNCTION)
def get_customer_info(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python Get Client Info function processed a request.')
    
    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        customer_id = req_body.get('customer_id')

    logging.info(f"Customer Id:  {customer_id}.")
    if not customer_id:
        return func.HttpResponse(
             "Function get_customer_info executed successfully. No Customer Id given!",
             status_code=200
        ) 
    try:
        conn = connect_db()
        response = db_get_customer_info(conn, customer_id)

        conn.close()
    except Exception as e:
        logging.critical("Error occured in get_customer_info: "+str(e))
        response = {"customer_id":customer_id, "status":"failure"}
    
    return func.HttpResponse(json.dumps(response, indent=4, sort_keys=True, default=str),mimetype="application/json",status_code=200)

@app.route(route="update_customer_info", auth_level=func.AuthLevel.FUNCTION)
def update_customer_info(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Update customer id processing a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        customer_id = req_body.get('customer_id')
        new_address = req_body.get('address')


    if not customer_id or not new_address:
        return func.HttpResponse(
             "Function update_customer_info executed successfully. No customer_id or address was given!",
             status_code=200
        ) 

    try:
        conn = connect_db()
        status = db_update_customer_address(conn, new_address, customer_id)

    
        response = {
            "customer_id":customer_id,
            "stauts": status
        }
    except:
        response = {
            "customer_id":-1,
            "stauts":"failure"
        }

    return func.HttpResponse(json.dumps(response, indent=4, sort_keys=True, default=str),mimetype="application/json",status_code=200)

