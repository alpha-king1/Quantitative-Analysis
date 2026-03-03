from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd


class StatTest:
    def __init__(self, df, structure_return, forward_return_column_name):
        '''
            This class is for getting stats of a structure against baseline
        :param df: This is the main data set and it must contain the following columns: 'forward_return' of any bar return you need, 'year' column that seperate the dataset by year, 'sessions', 'vol_regime'.

        :param structure_return: this is the dataset containing the forward return after a specific structure, it must have the following column: 'time', 'Return' this for a specific bar forward return we would be testing, 'sessions', 'vol_regime'.
        '''
        self.data = df
        self.structure_return = structure_return
        self.structure_return['year'] = self.structure_return['time'].dt.year
        self.forward_return = forward_return_column_name

    def ttest(self):
        t_stat, p_value = stats.ttest_ind(
            self.structure_return['Return'],
            self.data[self.forward_return],
            equal_var=False,
            nan_policy='omit'
        )
        return p_value

    def MW_test(self):
        stat, p = mannwhitneyu(self.structure_return['Return'].dropna(), self.data[self.forward_return].dropna(),
                               alternative='two-sided')
        return p

    def ks2_test(self):
        stat, p = ks_2samp(self.structure_return['Return'].dropna(), self.data[self.forward_return].dropna())
        return p

    def bootstrap_resampling(self):
        boot_means = []

        for _ in range(10000):
            sample = np.random.choice(self.structure_return['Return'].dropna(),
                                      size=len(self.structure_return['Return'].dropna()),
                                      replace=True)
            boot_means.append(np.mean(sample))

        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)
        return lower

    def yearly(self, column):
        result = []

        for year in sorted(self.data[column].unique()):

            yearly = self.data[self.data[column] == year]
            structure_year = self.structure_return[self.structure_return[column] == year]

            structure = structure_year['Return']
            baseline = yearly['forward_return_10bar']

            if len(structure) < 10:
                continue  # skip tiny samples

            struct_mean = structure.mean()
            base_mean = baseline.mean()
            diff = struct_mean - base_mean

            result.append({
                'year': year,
                'structure_mean': struct_mean,
                'baseline_mean': base_mean,
                'mean_diff': diff,
                'sample_size': len(structure)
            })
        yearly_result = pd.DataFrame(result)
        return yearly_result

    def column(self, column_name):
        vol_results = []

        for regime in self.data[column_name].unique():

            subset = self.data[self.data[column_name] == regime]
            subset_structure = self.structure_return[self.structure_return[column_name] == regime]

            structure = subset_structure['Return']
            baseline = subset['forward_return_10bar']

            if len(structure) < 10:
                continue

            struct_mean = structure.mean()
            base_mean = baseline.mean()
            diff = struct_mean - base_mean

            vol_results.append({
                'vol_regime': regime,
                'structure_mean': struct_mean,
                'baseline_mean': base_mean,
                'mean_diff': diff,
                'sample_size': len(structure)
            })

        result = pd.DataFrame(vol_results)
        return result

