import pickle
import pandas as pd
filename = 'svc_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
test_data= pd.read_csv("testDatasetExample.csv")
output_df = pd.DataFrame()
output_df["patient ID"] = test_data["ID"]
test_data.drop(columns=["ID"], inplace=True)
test_data = test_data.apply(pd.to_numeric)

cat_col=[
 'ER',
 'PgR',
 'HER2',
 'TrippleNegative',
 'ChemoGrade',
 'Proliferation',
 'HistologyType',
 'LNStatus',
 'TumourStage']

for i in test_data.columns:
    if i not in cat_col:
        test_data[i] = test_data[i].replace(999,test_data[i].mean())
    else:
        test_data[i] =  test_data[i].replace(999,test_data[i].mode()[0])

y_pred = loaded_model.predict(test_data)
output_df["Predicted PCR"] = y_pred
output_df.to_csv("FinalTestPCR_result.csv",index=False)

print(pickle.format_version)