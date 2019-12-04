import graphlab as gl 
import graphlab.aggregate as agg
import pandas as pd


ratings_df = gl.SFrame("dataset/train-PDA2019.csv")
final_file = pd.read_csv("dataset/test-PDA2019.csv")
users = final_file.iloc[:, 0]

# train the recommender model
recommender = gl.recommender.ranking_factorization_recommender.create(ratings_df, user_id = "userID", item_id = "itemID", target = "rating")

# predict top 10 items for the requested users
recs = recommender.recommend(users.tolist(), k = 10)

# process the result to output the correct csv file for the competition
for user in users:
	first10 = ' '.join(map(str, recs[recs["userID"]==user]["itemID"]))
	final_file.loc[final_file["userID"] == user, "recommended_itemIDs"] = first10 

final_file.to_csv("dataset/recommendations.csv", index = False)