# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone: Recipe Recommender

## Background & Problem Statement

Food waste is one of the biggest waste streams in Singapore and the amount of food waste generated has grown by around 20% over the last 10 years.

There are 3 ways to prevent food waste:
1. Buy, order, or cook what you can finish. Make a shopping list of things you need so you won’t overbuy.
2. Turn your leftovers or excess ingredients into new dishes instead of throwing them away
3. Donate excess food to organizations like Singapore Food Bank

In this project, I would like to address item no 2: The leftovers / excess ingredients. Although there are no lack of online recipes out there, often times recipe searching was done before you do your grocery shopping to buy the ingredients. But what if it is turned the other way round? Based on the available ingredients, what are the recipes we can make? To address this problem, a recipe recommendation system that would take users input for the ingredients, and produce an output of recipes that uses the ingredients were created.


## Datasets

The data is scrapped from [Epicurious](http://www.epicurious.com/recipes-menus) and uploaded to Kaggle:
1. [Epicurious_1](https://www.kaggle.com/datasets/hugodarwood/epirecipes) by Hugodarwood.
2. [Epicurious_2](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images) by Sakshi Goel.

Additionally, food ingredients list are also scrapped from:
1. [BBC Food](https://www.bbc.co.uk/food/ingredients/a-z/a/1)


## EDA, Data Cleaning & Pre-Processing
In order to build the recipe recommendation system, each ingredients in the ingredient list have to be extracted to remove redundant information that would not help distinguish recipes. For example:
- 'beets' has to be extracted from '3 pounds small beets, scrubbed thoroughly but not peeled'
- 'brown sugar' has to be extracted from '1 cup of brown sugar'

The first method is by using NLTK POS (Part of Speech) to extract all the nouns & adjective of each ingredients. At this point, 'small beets' would be extracted from '3 pounds small beets, scrubbed thoroughly but not peeled', and 'cup brown sugar' would be extracted from '1 cup of brown sugar'. Next, standard cooking measurements such as teaspoon, tablespoon, pounds, cups are removed. At this point, our ingredients will become 'small beets' and 'brown sugar'. Applying this to our dataframe, we ended up with 32k unique ingredients as we are unable to account for non-standard cooking measurements such as 'bulb' in '1 bulb of fennel', as well as adjectives such as 'fresh' in '1 cup of fresh chilli'.

The second method is by scrapping ingredients from BBC Food. 1200 unique food ingredients were extracted from the website. Firstly, ‘NLTK singularize’ were used to singularize all ingredients from recipe dataframe & bbc dataframe. Next, all ingredients in recipe dataframe were tokenized in Trigram, Bigram, Unigram (Trigram ingredient:  ‘all purpose flour’, Bigram ingredient: ‘brown sugar’, Unigram ingredient: ‘tomato’). Finally, we can match all the ingredient from original recipe dataframe with bbc dataframe.

As the second method (BBC Food) are more effective in extracting key ingredients, it is selected over the NLTK method.


## Modelling
To create the 'Search Recipe' feature, first we filter our dataframe based on the input ingredients. Next, we create a TF-IDF to generate embeddings for each ingredients. The IDF measures the importance of a term across the whole corpus. IDF will weigh down terms that are very common across a corpus (in our case words like salt, sugar, olive oil, butter, garlic, etc.) and weighs up rare terms. This is useful for us as it will give us better distinguishing power between recipes as the ingredients that are scaled-down will be ingredients that the user will tend to not give as an input to the recommendation system.

'You might also like' feature are build using content-based recommendation system which enables us to recommend recipes to people based on the original recipes that they selected. To measure the cosine-similarity between recipes, 'tag' features was used. The 'category' feature contains key information such as: the main ingredient of the recipe (i.e. Tomato, Egg), ocassion (i.e. New Year, Christmas, Birthday), the dietary information (i.e. Peanut-Free, Glutten-Free), as well as the timing (i.e. Breakfast, Lunch). An example of a 'category' for 'Lentil, Apple, and Turkey Wrap' is Kid-Friendly, Sandwich, Bean, Fruit, Tomato, turkey, Vegetable, Apple, Lentil, Lettuce.


## Summary & Limitation
Due to time constraint, the following are parked under future works:
- To deploy streamlit on clouds
- To incorporate the amount of leftover ingredients (i.e. 2 cucumber, 3 slices of bread)
- To incorporate local recipes & ingredients (Current recipes are US based)
- To explore on how to better extract key ingredients from the ingredient list (Model training, etc)

