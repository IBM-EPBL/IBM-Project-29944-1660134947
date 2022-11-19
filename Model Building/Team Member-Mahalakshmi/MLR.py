#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[2]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='0j9ybHNI1hQ6b2L9ZiJlLBRjwWf3tbtFoNAoTtZLrz9z',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'predictingenergyoutputofwindturbi-donotdelete-pr-vvuc1mxciqx0o2'
object_key = 'T1.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()


# In[3]:


df.rename(columns={"Date/Time":"Time",
                   "LV ActivePower (kW)":"ActivePower(KW)",
                   "Wind Speed (m/s)": "WindSpeed(m/s)",
                   "Wind Direction(°)":"Wind_Direction"},
                   inplace=True)


# In[4]:


df


# In[5]:


sns.pairplot(df)


# In[6]:


plt.figure(figsize=(10, 8))
corr = df.corr()
ax = sns.heatmap(corr, vmin = -1,vmax = 1,annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

print(corr)


# In[8]:


df["Time"] = pd.to_datetime(df["Time"], format = "%d %m %Y %H %M", errors = "coerce")


# In[9]:


y = df["ActivePower(KW)"]
X = df[["Theoretical_Power_Curve (KWh)", "WindSpeed(m/s)"]]

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


# ## Model building

# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

forest_model = RandomForestRegressor(n_estimators = 750, max_depth = 4, max_leaf_nodes = 500, random_state = 1)

forest_model.fit(train_X, train_y)


# In[13]:


RandomForestRegressor(max_depth=4, max_leaf_nodes=500, n_estimators=750,
                      random_state=1)


# In[14]:


power_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, power_preds))
print(r2_score(val_y, power_preds))


# In[15]:


joblib.dump(forest_model, "power_prediction.sav")


# In[16]:


df


# In[17]:


import pickle
pickle.dump(forest_model,open("model.pkl","wb"))


# In[18]:


get_ipython().system('pip install -U ibm-watson-machine-learning')


# In[19]:


from ibm_watson_machine_learning import APIClient
import json 
import numpy as np


# In[20]:


wml_credentials = {
    "apikey":"Ilik-kKvZ4Lwtruh-l7Bl2FS5IEFz0Ujhq8533qqw09Z",
    "url":"https://us-south.ml.cloud.ibm.com"
}


# In[25]:


wml_client = APIClient(wml_credentials)


# In[27]:


wml_client.spaces.list() 


# In[28]:


SPACE_ID = "f1f8ff94-56cd-4e00-a0c9-f8ae5161c050"


# In[29]:


wml_client.set.default_space(SPACE_ID)


# In[30]:


wml_client.software_specifications.list()


# # save and Deploy the model

# In[31]:


import sklearn
sklearn.__version__


# In[32]:


MODEL_NAME = 'DemoModel_MLR'
DEPLOYMENT_NAME = 'Wind turbine'
DEMO_MODEL = forest_model


# In[33]:


software_spec_uid = wml_client.software_specifications.get_id_by_name('runtime-22.1-py3.9')


# In[34]:


# setup model meta
model_props ={
    wml_client.repository.ModelMetaNames.NAME: MODEL_NAME,
    wml_client.repository.ModelMetaNames.TYPE:'scikit-learn_1.0',
    wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
    
}


# In[35]:


#save model
model_details = wml_client.repository.store_model(
    model=DEMO_MODEL,
    meta_props=model_props,
    training_data= train_X,
    training_target= train_y
)


# In[36]:


model_details


# In[37]:


model_id = wml_client.repository.get_model_id(model_details)


# In[38]:


model_id


# In[39]:


#deploy in purpose
deployment_props = {
     wml_client.deployments.ConfigurationMetaNames.NAME:DEPLOYMENT_NAME,
     wml_client.deployments.ConfigurationMetaNames.ONLINE:{}
}


# In[40]:


deployment = wml_client.deployments.create(
    artifact_uid=model_id,
    meta_props=deployment_props
)


# In[ ]:




