import dask.dataframe as dd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
ddf = dd.read_parquet('data/card_transactions/*.parquet')
ddf = ddf.assign(client_id=ddf["client_id"].astype("string"))
ddf = ddf.assign(
    is_weekend = ddf["day_of_week"].isin([6, 7]).astype(int)
)
sings = ddf.groupby('client_id').agg(
    avg_transactions   = ('amnt', 'mean'),
    sum_transactions   = ('amnt', 'sum'),
    count_transactions = ('client_id', 'count'),
    ecommerce_share    = ('ecommerce_flag', 'mean'),
    income_share = ('income_flag', 'mean'),
    hour_diff_mean     = ('hour_diff', 'mean'),
    weekend_share      = ('is_weekend', 'mean'),
).reset_index()


nunique_cols = ["currency", "operation_type", "card_type", "operation_type_group",
                "payment_system", 'mcc', 'country', 'city', "weekofyear"]

nunique_parts = [
    ddf.groupby("client_id")[col].nunique().rename(f"n_{col}")
    for col in nunique_cols
]

nunique_df = dd.concat(nunique_parts, axis=1).reset_index()


sings_pd   = sings.compute()
nunique_pd = nunique_df.compute()

result = sings_pd.merge(nunique_pd, on="client_id", how="left")
result["tx_per_week"] = result["count_transactions"] / result["n_weekofyear"]
print(result.head())

scaler_cols = [i for i in result.columns if i != 'client_id']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(result[scaler_cols])

print(X_scaled.shape)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))


kmeans_final = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10,
)


labels = kmeans_final.fit_predict(X_scaled)


result["cluster"] = labels

print(result[["client_id", "cluster"]].head())
print("\nРазмеры кластеров:")
print(result["cluster"].value_counts().sort_index())

feature_cols = [c for c in result.columns if c not in ("client_id", "cluster")]


plot_df = result.sample(n=50000, random_state=42)


for col in feature_cols:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(
        data=plot_df,
        x=col,
        hue="cluster",
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
        ax=axes[0]
    )
    axes[0].set_title(f"Histplot: {col}")

    for cluster_id in sorted(plot_df['cluster'].unique()):
        subset = plot_df[plot_df['cluster'] == cluster_id]
        sns.kdeplot(
            data=subset,
            x=col,
            label=f"Кластер {cluster_id}",
            fill=True,
            alpha=0.3,
            ax=axes[1]
        )
    axes[1].set_title(f"KDE: {col}")
    axes[1].legend(title="Кластер")

    plt.tight_layout()
    plt.show()

print(result.groupby('cluster')[feature_cols].mean().T.round(2))