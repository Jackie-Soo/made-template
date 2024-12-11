import unittest
from unittest.mock import patch, Mock
import os
import pandas as pd
from project3 import main, download_dataset, read_csv, transform_crime_enforcement_data, transform_factors_data, load_to_db

TEST_DATA_DIR = '~/.kaggle'  # path of the downloaded datasets
DATABASE_PATH = os.path.join('data', 'project3.db')  # path of the output file


class SystemTestDataPipeline(unittest.TestCase): # System-test level

    @patch('project3.kagglehub.dataset_download')
    @patch('project3.read_csv')
    @patch('project3.sqlite3.connect')
    def test_full_data_pipeline(self, mock_connect, mock_read_csv, mock_download_dataset):
        # data downloading
        mock_download_dataset.side_effect = [
            os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/fbi-us/california-crime/versions/1'),  # download path1
            os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1')  # download path2
        ]

        # read csv
        mock_read_csv.side_effect = [
            pd.read_csv(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/fbi-us/california-crime/versions/1', 'ca_offenses_by_city.csv'), encoding='UTF-8'),
            pd.read_csv(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/fbi-us/california-crime/versions/1', 'ca_law_enforcement_by_city.csv'), encoding='UTF-8'),
            pd.read_csv(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1', 'MedianHouseholdIncome2015.csv'), encoding='ISO-8859-1'),
            pd.read_csv(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1', 'PercentagePeopleBelowPovertyLevel.csv'), encoding='ISO-8859-1'),
            pd.read_csv(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1', 'PercentOver25CompletedHighSchool.csv'), encoding='ISO-8859-1')
        ]

        print(mock_read_csv.call_args_list) # check the path of 5 datasets

        # connect the database
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # run main (project3.py)
        main()

        # validata if the datasets are downloaded
        mock_download_dataset.assert_any_call("~/.kaggle/kagglehub/datasets/fbi-us/california-crime/versions/1")
        mock_download_dataset.assert_any_call("~/.kaggle/kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1")

        # validate if the datasets are read
        mock_read_csv.assert_any_call(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/fbi-us/california-crime/versions/1', 'ca_offenses_by_city.csv'))
        mock_read_csv.assert_any_call(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/fbi-us/california-crime/versions/1', 'ca_law_enforcement_by_city.csv'))
        mock_read_csv.assert_any_call(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1', 'MedianHouseholdIncome2015.csv'))
        mock_read_csv.assert_any_call(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1', 'PercentagePeopleBelowPovertyLevel.csv'))
        mock_read_csv.assert_any_call(os.path.join(TEST_DATA_DIR, 'kagglehub/datasets/kwullum/fatal-police-shootings-in-the-us/versions/1', 'PercentOver25CompletedHighSchool.csv'))


        c_data = mock_read_csv.side_effect[0]  # 'ca_offenses_by_city.csv'
        e_data = mock_read_csv.side_effect[1]  # 'ca_law_enforcement_by_city.csv'
        i_data = mock_read_csv.side_effect[2]  # 'MedianHouseholdIncome2015.csv'
        p_data = mock_read_csv.side_effect[3]  # 'PercentagePeopleBelowPovertyLevel.csv'
        h_data = mock_read_csv.side_effect[4]  # 'PercentOver25CompletedHighSchool.csv'

        # test the transformed data
        transformed_data = transform_crime_enforcement_data(c_data, e_data)
        transformed_factors_data = transform_factors_data(i_data, p_data, h_data)

        expected_columns = [
            'City', 'Violent crime', 'Murder and nonnegligent manslaughter',
            'Rape (revised definition)', 'Robbery', 'Aggravated assault',
            'Property crime', 'Burglary', 'Larceny-theft', 'Motor vehicle theft',
            'Arson', 'Population', 'Total law enforcement employees',
            'Total officers', 'Total civilians'
        ]
        # validate column names
        self.assertListEqual(list(transformed_data.columns), expected_columns)

        expected_dtypes = {
            'City': 'string',
            'Violent crime': 'int32',
            'Murder and nonnegligent manslaughter': 'int32',
            'Rape (revised definition)': 'int32',
            'Robbery': 'int32',
            'Aggravated assault': 'int32',
            'Property crime': 'int32',
            'Burglary': 'int32',
            'Larceny-theft': 'int32',
            'Motor vehicle theft': 'int32',
            'Arson': 'int32',
            'Population': 'int32',
            'Total law enforcement employees': 'int32',
            'Total officers': 'int32',
            'Total civilians': 'int32'
        }
        # validate datatype
        for column, dtype in expected_dtypes.items():
            with self.subTest(column=column):
                self.assertEqual(transformed_data[column].dtype.name, dtype,
                                 f"Column {column} dtype doesn't match")


        expected_columns_factors = [
            'City', 'Median Income', 'poverty_rate', 'percent_completed_hs'
        ]
        # column names validation
        self.assertListEqual(list(transformed_factors_data.columns), expected_columns_factors)

        # expected datatype
        expected_dtypes_factors = {
            'City': 'string',
            'Median Income': 'int32',
            'poverty_rate': 'float64',
            'percent_completed_hs': 'float64'
        }
        # datatype validation
        for column, dtype in expected_dtypes_factors.items():
            with self.subTest(column=column):
                self.assertEqual(transformed_factors_data[column].dtype.name, dtype,
                                 f"Column {column} dtype does not match")

        # validate if data is loaded into the dataset
        mock_conn.execute.assert_any_call(
            "CREATE TABLE crime_enforcement ("
            "City STRING, "
            "Violent crime INT, "
            "Murder and nonnegligent manslaughter INT, "
            "Rape (revised definition) INT, "
            "Robbery INT, "
            "Aggravated assault INT, "
            "Property crime INT, "
            "Burglary INT, "
            "Larceny-theft INT, "
            "Motor vehicle theft INT, "
            "Arson INT, "
            "Population INT, "
            "Total law enforcement employees INT, "
            "Total officers INT, "
            "Total civilians INT);"
        )

        mock_conn.execute.assert_any_call(
            "CREATE TABLE factors ("
            "City STRING, "
            "Median Income INT, "
            "poverty_rate FLOAT, "
            "percent_completed_hs FLOAT);"
        )

        # validate if the data is inserted into the dataset
        mock_conn.execute.assert_any_call(
            "INSERT INTO crime_enforcement (City, Violent crime, Murder and nonnegligent manslaughter, "
            "Rape (revised definition), Robbery, Aggravated assault, Property crime, Burglary, "
            "Larceny-theft, Motor vehicle theft, Arson, Population, Total law enforcement employees, "
            "Total officers, Total civilians) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )

        mock_conn.execute.assert_any_call(
            "INSERT INTO factors (City, Median Income, poverty_rate, percent_completed_hs) VALUES (?, ?, ?, ?)"
        )

        # transaction validation, to see if it committed successfully
        mock_conn.commit.assert_called_once()
        # disconnection validation
        mock_conn.close.assert_called_once()

        # validate if the logging file exists
        with open('project3.log', 'r') as log_file:
            log_content = log_file.read()
            self.assertIn('Successfully download Dataset', log_content)
            self.assertIn('Successfully read CSV file', log_content)
            self.assertIn('Successfully transform crime&enforcement data', log_content)
            self.assertIn('Successfully load data to table', log_content)

        # Verify if the database file exists after running the pipeline
        self.assertTrue(os.path.exists(DATABASE_PATH),
                        "file 'project3.db' should exist in the 'data' directory.")


if __name__ == '__main__':
    unittest.main()
