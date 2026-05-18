import pandas as pd

# CLUSTER IMPROVED 2 CLUSTER IMPROVEDDAN DAHA IYI SANKI.
# TEK TEK ORNEKLERE BAKMADIM AMA TAGLERE DAHA DETAYLI BAKIYOR. TAG FALLBAK KODUNDAN GELIYOR
#HT T7 U DA SUANDA CLLUSTER IMPROVEDDAKILER VAR


df2 = pd.read_csv("7-CLUSTERING_MODELS/clusters_improved_2/family_assignments.csv")


pd.set_option("display.max_rows", None)

counts = df2["family_root"].value_counts()
print(counts)