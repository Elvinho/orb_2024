import pandas as pd

dic_aux = {"date":[1,2,3],"y":["a","b","c"]}
df = pd.DataFrame(dic_aux)

print(df["y"].iloc[-1])