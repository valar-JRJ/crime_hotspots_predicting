from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from utils import combine_csv


# aggregating and cleaning the results
results = combine_csv('./results/*/')
results = results.groupby(['type','month', 'Model', 'baseline', 'measure'], as_index=False).max()
results = results.groupby(['type', 'Model', 'baseline', 'measure'], as_index=False).mean().drop(columns=['month'])
crime_type = {'drugs': 'Drugs', 'weapons': 'Possession of weapons', 'violence': 'Violence and sexual offences'}
results = results.replace({'kde': 'KDE', 'imd': 'IMD', 'mob': 'Mobility', 'full': 'Full', 'no-kde': 'IMD+Mobility',
                           'no-imd': 'KDE+Mobility', 'no-mob': 'KDE+IMD'})

# extract results for each crime type
drugs = results.loc[results['type'] == crime_type['drugs']]
weapons = results.loc[results['type'] == crime_type['weapons']]
violence = results.loc[results['type'] == crime_type['violence']]

# plot
od = ['KDE', 'IMD', 'Mobility', 'KDE+Mobility', 'KDE+IMD', 'Full']
sns.set_theme(style="whitegrid")

# barplot for drugs
g = sns.catplot(x="measure", y="score", hue="baseline", col="Model", sharey=False, sharex=False, hue_order=od,
                data=drugs, kind="bar", col_wrap=3, height=4, margin_titles=False)
g.set_xticklabels(['Accuracy', 'AUC', 'F1']).set_axis_labels('', '')
sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
g.set(ylim=(0.0,1.0))
# plt.savefig('fig/drugs_result.png', bbox_inches='tight', dpi=200)
plt.show()

# barplot for possession of weapons
g = sns.catplot(x="measure", y="score", hue="baseline", col="Model", sharey=False, sharex=False, hue_order=od,
                data=weapons, kind="bar", col_wrap=3, height=4, margin_titles=False)
g.set_xticklabels(['Accuracy', 'AUC', 'F1']).set_axis_labels('', '')
sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
g.set(ylim=(0.0,1.0))
# plt.savefig('fig/weapons_result.png', bbox_inches='tight', dpi=200)
plt.show()

# barplot for violence and sexual offences
g = sns.catplot(x="measure", y="score", hue="baseline", col="Model", sharey=False, sharex=False, hue_order=od,
                data=violence, kind="bar", col_wrap=3, height=4, margin_titles=False)
g.set_xticklabels(['Accuracy', 'AUC', 'F1']).set_axis_labels('', '')
sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
g.set(ylim=(0.0,1.0))
# plt.savefig('fig/violence_result.png', bbox_inches='tight', dpi=200)
plt.show()

# comparison between models with full features
fulls = results.loc[results['baseline']=='Full'].sort_values(by=['type', 'measure', 'score'],
                                                             ascending=[True, True, False])
g = sns.catplot(data=fulls, x='score', y='type', hue='Model', col='measure', sharey=True, sharex=False, kind="bar")
sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False)
g.set(xlim=(0.0, 1.0))
# plt.savefig('fig/algorithms.png', bbox_inches='tight', dpi=200)
plt.show()

# comparison between different types of crimes
diff_types = results.loc[(results['baseline'].isin(['KDE', 'IMD', 'Mobility', 'Full']))&(results['measure'] == 'f1')]
g = sns.catplot(data=diff_types, x='Model', y='score', hue='type', col='baseline',
                col_order=['Full', 'KDE', 'IMD', 'Mobility'], sharey=False, sharex=False, kind="bar")
sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False)
g.set_titles('{col_name}')
g.set(ylim=(0.0, 1.0))
# plt.savefig('fig/crime_type_acc.png', bbox_inches='tight', dpi=200)
plt.show()
