from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt


data = load_breast_cancer(as_frame=True)
df = data.frame
df['target_name'] = df['target'].map(dict(enumerate(data.target_names)))
plt.figure()
groups = [df[df['target_name']=='malignant']['mean radius'],
          df[df['target_name']=='benign']['mean radius']]
plt.boxplot(groups, labels=['malignant','benign'])
plt.ylabel("Mean radius")
plt.title("Breast Cancer: Mean radius by class")
plt.tight_layout()
plt.show()
