import mysql.connector
from mysql.connector import Error


def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='plant_monitoring'
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def initialize_database():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    observation_date DATE NOT NULL,
                    observation_time TIME NOT NULL,
                    leaf_image_path VARCHAR(255) NOT NULL,
                    dimension_image_path VARCHAR(255) NOT NULL,
                    leaf_count INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.commit()
            print("Database initialized successfully")
        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()


def save_observation(observation_date, observation_time, leaf_image_path, dimension_image_path, leaf_count):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()

            query = """
                INSERT INTO observations 
                (observation_date, observation_time, leaf_image_path, dimension_image_path, leaf_count)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query,
                           (observation_date, observation_time, leaf_image_path, dimension_image_path, leaf_count))

            connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Error saving observation: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()


def get_all_observations():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)

            cursor.execute("""
                SELECT id, observation_date, observation_time, leaf_count
                FROM observations
                ORDER BY observation_date ASC, observation_time ASC
            """)

            return cursor.fetchall()
        except Error as e:
            print(f"Error fetching observations: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()


if __name__ == "__main__":
    initialize_database()