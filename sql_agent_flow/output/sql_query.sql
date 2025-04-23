```sql
SELECT sale_date, SUM(quantity_sold) AS total_quantity_sold
FROM Sales
WHERE product_id = (SELECT product_id FROM Product WHERE product_name = 'C')
GROUP BY sale_date
ORDER BY sale_date;
```