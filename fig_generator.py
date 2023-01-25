import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py


# read iclr2017
df = pd.read_json('datasets/iclr_2017.json')


scores = df['recommendation']
scores = [ np.mean(score) for score in scores]

df['scores'] = scores
# drop rows with scores less than 4
df = df[df['scores'] > 4]

# accepted paper
accepted = df[df['accepted'] == True]
# rejected paper
rejected = df[df['accepted'] == False]

# drop the outlier citation papers with 5 percent for accepted and rejected papers
accepted_norm = accepted[accepted['citation'] < accepted['citation'].quantile(0.95)]
rejected_norm = rejected[rejected['citation'] < rejected['citation'].quantile(0.95)]


bins = np.arange(5, 9.5, 0.5)

plt.figure(figsize=(7, 5))
plt.xlabel('Recommendation Score')
plt.ylabel('Count Per Bin')
plt.title('Review Score Histogram')

plt.hist(accepted['scores'], bins=bins, color='#313369', label='All Accepted' ,alpha = 0.5)

plt.hist(accepted_norm['scores'], bins=bins, alpha = 0.5, color='#6ec1ff', label='95 Quantile Accepted', edgecolor='black')
plt.savefig('figures/accepted_score_hist.pdf')
plt.legend()