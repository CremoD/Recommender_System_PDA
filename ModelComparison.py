import graphlab as gl 
import graphlab.aggregate as agg
import pandas as pd


ratings_df = gl.SFrame("dataset/train-PDA2019.csv")
item_df = gl.SFrame("dataset/content-PDA2019.csv")

final_file = pd.read_csv("dataset/test-PDA2019.csv")
users = final_file.iloc[:, 0]

train, test = gl.recommender.util.random_split_by_user(ratings_df, user_id = "userID", item_id = "itemID")

############## TRY COMPARISON ##############
recommender1 = gl.recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating", item_data = item_df) 
recommender2 = gl.recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating")
recommender3 = gl.recommender.factorization_recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating") 
recommender4 = gl.recommender.factorization_recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating", item_data = item_df) 
recommender5 = gl.recommender.ranking_factorization_recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating") 
recommender6 = gl.recommender.ranking_factorization_recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating", item_data= item_df) 

# Try factorization recommender with different parameters
recommender_fact_1 = gl.recommender.ranking_factorization_recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating", num_factors = 40) 
recommender_fact_2 = gl.recommender.ranking_factorization_recommender.create(train, user_id = "userID", item_id = "itemID", target = "rating", num_factors = 28) 

gl.recommender.util.compare_models(test, [recommender1, recommender2,recommender3,recommender4,recommender5,recommender6, recommender_fact_1, recommender_fact_2], model_names=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"])
