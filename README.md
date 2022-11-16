# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone: Recipe Recommender

## Background & Problem Statement

Food waste accounts for about 12 per cent of the total waste generated in Singapore. Therefore, there is a need to manage food waste holistically. Reducing food wastage, redistributing unsold or excess food, and recycling/treating food waste are important components of our national waste management strategies to work towards Singapore becoming a Zero Waste Nation.

Due to the pandemic, many of us have settled into new routines that involve a lot more working from home & home cooking. However, most of the time we will end up accumulating unused ingredients. If remain unused, these ingredients turn into food wastage.To address this problem, a recipe recommendation system that would take users input for the ingredients, and produce an output of recipes that uses the ingredients were created.


## Datasets

The data is scrapped from [Epicurious](http://www.epicurious.com/recipes-menus) and uploaded to Kaggle:
1. [Epicurious_1](https://www.kaggle.com/datasets/hugodarwood/epirecipes) by Sakshi Goel.
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

