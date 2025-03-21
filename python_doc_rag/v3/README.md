# STEPS TO RUN:

> change directory to python_doc_rag <br>
 create .env file similar to .env.sample<br>
 add GOOGLE_API_KEY key in .env file in backend <br>


```python
>> uvicorn v3.backend.main:app --reload
>> python -m streamlit run v3/frontend/main.py
```
