from typing import Any, Type, Union
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class NL2SQLToolInput(BaseModel):
    query: str = Field(
        title="SQL Query",
        description="The SQL query to execute.",
    )


class NL2SQLTool(BaseTool):
    name: str = "NL2SQLTool"
    description: str = "Converts natural language to SQL queries and executes them."
    db_uri: str = Field(
        title="Database URI",
        description="The URI of the SQLite database to connect to.",
    )
    tables: list = []
    columns: dict = {}
    args_schema: Type[BaseModel] = NL2SQLToolInput

    def model_post_init(self, __context: Any) -> None:
        """Initialize the tool by fetching table and column metadata."""
        data = {}
        tables = self._fetch_available_tables()
        for table in tables:
            table_columns = self._fetch_all_available_columns(table["table_name"])
            data[f'{table["table_name"]}_columns'] = [
                {"column_name": row["name"], "data_type": row["type"]}
                for row in table_columns
            ]

        self.tables = tables
        self.columns = data

    def _fetch_available_tables(self):
        """Fetch all table names in an SQLite database."""
        return self.execute_sql(
            "SELECT name AS table_name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )

    def _fetch_all_available_columns(self, table_name: str):
        """Fetch column names and data types for a given table in SQLite."""
        return self.execute_sql(f"PRAGMA table_info({table_name});")

    def _run(self, query: str):
        """Execute an SQL query and return the results."""
        try:
            data = self.execute_sql(query)
        except Exception as exc:
            data = (
                f"Based on these tables {self.tables} and columns {self.columns}, "
                "you can create SQL queries to retrieve data from the database. "
                f"Original request: {query}. Error: {exc}. Please correct the SQL query."
            )

        return data

    def execute_sql(self, query: str) -> Union[list, str]:
        """Executes an SQL query and returns results as a list of dictionaries or a success message."""
        engine = create_engine(self.db_uri, connect_args={"check_same_thread": False})
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            result = session.execute(text(query))
            session.commit()

            if result.returns_rows:
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                return data
            else:
                return f"Query '{query}' executed successfully"

        except Exception as e:
            session.rollback()
            raise e

        finally:
            session.close()
