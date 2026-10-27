import json
import os
from abc import ABC, abstractmethod

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper


class Connector(ABC):
    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        pass


class GraphDBConnector(Connector):
    def __init__(self, path_to_credentials):
        """
        Open a connection to GraphDB
        Files needed:
            - secrets/graphdb.txt : server_name, repository_name, format
        """
        assert path_to_credentials.endswith(
            ".json"
        ), f"Expected a path to a .json file for credentials, but got: {path_to_credentials}"

        with open(path_to_credentials) as config_file:
            config = json.load(config_file)
            server_name = config["server_name"]
            repository_name = config["repository_name"]
            format = config["format"]

        url = server_name + repository_name
        self.conn = SPARQLWrapper(url)

        # Set the resulting format as JSON
        if format == "JSON":
            self.conn.setReturnFormat(JSON)
        else:
            raise NotImplementedError

    def query(self, query: str) -> pd.DataFrame:
        self.conn.setQuery(query)
        return pd.DataFrame(self._process_query(self.conn.query().convert()))

    def _process_query(self, results):
        column_names = results["head"]["vars"]
        bindings = results["results"]["bindings"]

        data = []

        for result in bindings:
            row = {}
            for name in column_names:
                try:
                    row[name] = result[name]["value"].strip("-")
                except:
                    row[name] = None
            data.append(row)

        return data

    def read_queries(self, path_to_queries):
        # Reading query files
        queries = {}

        for filename in os.listdir(path_to_queries):
            if filename.endswith(".txt"):
                # Get the full path to the file
                file_path = os.path.join(path_to_queries, filename)

                # Open the file and read the query
                with open(file_path, "r", encoding="utf-8") as file:
                    query_content = file.read()

                # Store the query content in the dictionary, using the filename
                query_name = os.path.splitext(filename)[0]
                queries[query_name] = query_content

        return queries
