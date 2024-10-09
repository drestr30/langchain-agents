import os
from dotenv import load_dotenv
from psycopg2.extensions import connection
import psycopg2

load_dotenv()

def query_to_list(query, args=(), one=False):
    print('running query_to_list ...')

    conn = connect_db()
    cur = conn.cursor()

    cur.execute(query, args)
    r = [dict((cur.description[i][0], value) \
               for i, value in enumerate(row)) for row in cur.fetchall()]

    conn.close()
    return (r[0] if r else None) if one else r

def connect_db()-> connection:
    POSTGRES_REMOTE_ENDPOINT = os.environ['POSTGRES_REMOTE_ENDPOINT']
    POSTGRES_REMOTE_USER = os.environ['POSTGRES_REMOTE_USER']
    POSTGRES_REMOTE_PASSWORD = os.environ['POSTGRES_REMOTE_PASSWORD']
    POSTGRES_DB_NAME = os.environ['POSTGRES_DB_NAME']
    sslmode = os.environ['POSTGRES_SSL_MODE']
    # logging.info(f"Env: {POSTGRES_REMOTE_ENDPOINT},{POSTGRES_DB_NAME},{POSTGRES_REMOTE_USER}")
    conn_string = f"host={POSTGRES_REMOTE_ENDPOINT} user={POSTGRES_REMOTE_USER} dbname={POSTGRES_DB_NAME} password={POSTGRES_REMOTE_PASSWORD} sslmode={sslmode}"
    print(conn_string)
    conn: connection = psycopg2.connect(conn_string)

    print("Postgres Connection established")
    return conn

def db_get_customer_info(conn, customer_id:str)-> str:
    # conn = connect_db()
    try:
        with conn.cursor() as cursor:
            query = """
                SELECT * FROM customer
                WHERE customer_id = %s
            """ 
            # cursor.execute(query, (customer_email,))
            # result = cursor.fetchone()
            result = query_to_list(query, (customer_id,))
            if result:
                return result[0]
            else:
                return 'Customer not found'
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    finally:
        conn.close()

def db_update_customer_address(conn, new_address, customer_id) -> str:
    msg = "success"
    # conn = connect_db()
    try:
        with conn.cursor() as cursor:
            query = """
               UPDATE customer
                SET customer_address = %s
                WHERE customer_id = %s;
            """ 
            cursor.execute(query, (new_address, customer_id))
            # result = cursor.fetchone()
            conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        msg = str(e)
    finally:
        conn.close()
    return msg


if __name__ == "__main__":
    res = get_customer_info(1)
    print(res['customer_email'])
