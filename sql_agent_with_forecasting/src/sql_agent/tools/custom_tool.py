from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    InfoSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from crewai.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase

db = SQLDatabase.from_uri("postgresql://postgres:Mayur%40356@localhost:5432/DVDRentals")


@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database."""
    return {"tables": ListSQLDatabaseTool(db=db).invoke("")}


@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input: a comma-separated list of tables.
    Output: the schema and sample rows for those tables.
    """
    return InfoSQLDatabaseTool(db=db).invoke(tables)


@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database and return the result."""
    return QuerySQLDatabaseTool(db=db).invoke(sql_query)
