# Fraud model

Script to download data, detect fraud users and upload result to Google Cloud bucket

# Example usage

```
python fraud_model.py --b_name "gs://e5e92660-4647-4f30-8c2c-4120a0aeb37f" --u_path "users/2017/09/09" 
--t_path "tracks/2017/09/09" --s_path "streams/2017/09/09/allcountries"
```

# Parameters specification

<ul>
<li>**--b_name** - google cloud bucket url</li>
<li>**--u_path** - path to file with users data on bucket</li>
<li>**--t_path** - path to file with tracks data on bucket</li>
<li>**--s_path** - path to file with streams data on bucket</li>
<li>**--std_ratio** - threshold in standard deviation scale to select outliers, default value is **2**</li>
<li>**--n_duplicates** - threshold to select users with specific number of possible duplicted accounts, default value is **5**</li>
<li>**--radius** - neighborhood radius at which we look for outliers, default value is **0.01**</li>
<li>**--train** - option to choose either we need to train fraud mode on current data or not, default value is **False**</li>
</ul>
