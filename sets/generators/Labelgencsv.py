import os
import pandas as pd
 
# delimiter = ','
#change depending on the subfolder
folder_path = 'noprov'
i=0
for _root, _dirs, files in os.walk(folder_path):
    for filepath in files:
        i+=1
        if filepath.endswith('.py'):
            df_dict = {'Generator': [0]}
            df =  pd.DataFrame(df_dict)
            df_filepath = 'label_' + filepath + '.csv'
            df.to_csv(df_filepath, sep = ',', index = False)
            print(i, "construido csv en", filepath)