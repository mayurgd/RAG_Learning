# STEPS TO RUN:

> change directory to python_doc_rag <br>
 create .env file similar to .env.sample<br>
 add GOOGLE_API_KEY key in .env file in backend <br>


```python
>> uvicorn v2.backend.main:app --reload
>> python -m streamlit run v2/frontend/main.py
```
