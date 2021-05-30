import pandas as pd 
from matplotlib import pyplot as plt 

stock_data = pd.read_csv("stock_data.csv")

company_lists = set(stock_data["Stock"])
print(company_lists)