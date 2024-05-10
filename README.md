# Machine Learning tests

A collection of algorithms that might be useful soon for different things : transaction monitoring, onboarding...

## Logistic Regression

Understanding : https://fr.wikipedia.org/wiki/Régression_logistique

Possible usages :
* Mimic Hawk:AI rules
  * Backup if Hawk:AI shuts down
  * Base to replicate our transaction monitoring with custom rules
    * Scores on February transactions based on January training :
    * 97% accuracy, 85% precision, 97% recall

## Clustering

Understanding : https://fr.wikipedia.org/wiki/Partitionnement_de_données

Possible usages :
* Define different clusters of clients based on how they use their accounts
  * Outliers research for the Compliance team, to find possible frauds
    * With recents tests, and based on a list of previous fraudulous accounts :
    * 70% of those customers are in the same cluster
    * 90% of those customers are in 3 clusters
  * Could be used by Management and Marketing team, to follow the impact of campaigns on customers habits
  
Will be plugged to a basic Django app before the Green-Got admin panel.
