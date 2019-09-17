import mysql.connector
from mysql.connector import Error


class SlouchDataAccess:
    def __init__(self):
        self.my_db = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="root"
        )

        self.cursor = self.my_db.cursor()

    def save_slouch(self, filename, slouch_list_data):
        """
        :param filename: Filename of the slouch list.
        :param slouch_list_data: List of the slouch and time.
        :return: No return value.
        """

        try:
            for slouch_data in slouch_list_data:
                self.cursor.callproc("test.usp_save_slouch", (filename, str(slouch_data["slouch"]), str(slouch_data["straight"]),
                                                                    float(slouch_data["head_tilted"]), float(slouch_data["head_straight"]),
                                                                    float(slouch_data["time"])))

            self.my_db.commit()

        except Error as e:
            print(e)
        finally:
            self.cursor.close()
            self.my_db.close()
