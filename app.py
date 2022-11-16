import streamlit as st
import numpy as np
import pickle
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
#from pattern.text.en import singularize
import webbrowser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title = "Kiwi Kitchen", page_icon = "🥝")

tfidf = TfidfVectorizer()


#---------------FUNCTIONS---------------#
def get_recommendations(title, is_vegetarian, is_peanut_free, is_gluten_free, is_low_calories, is_kid_friendly):
    global recipe_1
    
    if is_vegetarian:
        recipe_1 = recipe_1[recipe_1['vegetarian'] == 1]
    if is_vegetarian:
        recipe_1 = recipe_1[recipe_1['peanut_free'] == 1]
    if is_vegetarian:
        recipe_1 = recipe_1[recipe_1['gluten_free'] == 1]
    if is_vegetarian:
        recipe_1 = recipe_1[recipe_1['low_calories'] == 1]
    if is_vegetarian:
        recipe_1 = recipe_1[recipe_1['kid_friendly'] == 1]
    
    recipe_1.reset_index(drop = True, inplace = True)
    
    tfidf_categories = pd.DataFrame(tfidf.fit_transform(recipe_1['tags_str']).toarray(),
                                    columns = tfidf.get_feature_names_out(),
                                    index = recipe_1['title'])

    cosine_sim = linear_kernel(tfidf_categories, tfidf_categories)
    recipe_index = recipe_1[recipe_1['title'] == title].index[0]
    distances = cosine_sim[recipe_index]
    recipe_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:4]
    
    similar_recipe_name = []
    similar_recipe_image = []
    for x in recipe_list:
        similar_recipe_name.append(recipe_1.iloc[x[0]]['title'])
        similar_recipe_image.append(recipe_1.iloc[x[0]]['Image_Name'])
    return similar_recipe_name, similar_recipe_image

def search_recipes(ingredient, is_vegetarian, is_peanut_free, is_gluten_free, is_low_calories, is_kid_friendly):
    global recipe
    recipe_2 = recipe.copy()
    
    if ingredient == []:
        # filter recipe based on dietary preferences
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['vegetarian'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['peanut_free'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['gluten_free'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['low_calories'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['kid_friendly'] == 1]
        
        from datetime import datetime
        hour = int(datetime.now().strftime('%H'))
        recipe_time = recipe_2.copy()
        
        if (hour >= 6) & (hour < 10):
            st.write("You have not selected any ingredients. Recommending you a few breakfast recipes!")
            recipe_time = recipe_2[recipe_2['breakfast'] == 1]
            
        elif (hour >= 10) & (hour < 11):
            st.write("You have not selected any ingredients. Recommending you a few brunch recipes!")
            recipe_time = recipe_2[recipe_2['brunch'] == 1]
            
        elif (hour >= 11) & (hour < 13):
            st.write("You have not selected any ingredients. Recommending you a few lunch recipes!")
            recipe_time = recipe_2[recipe_2['lunch'] == 1]

        elif (hour >= 13) & (hour < 17):
            st.write("You have not selected any ingredients. Recommending you a few dessert recipes!")
            recipe_time = recipe_2[recipe_2['dessert'] == 1]

        elif (hour >= 17) & (hour < 22):
            st.write("You have not selected any ingredients. Recommending you a few dinner recipes!")
            recipe_time = recipe_2[recipe_2['dinner'] == 1]

        elif (hour >= 22) | (hour < 6):
            st.write("You have not selected any ingredients. Recommending you a few supper recipes!")
            recipe_time = recipe_2[recipe_2['dessert'] == 1]
        
        recipe_time.reset_index(drop = True, inplace = True)
        recipe_index_list = recipe_time[['title']].index.tolist()
        recipe_random_list = random.sample(recipe_index_list, 3)
        
        # return
        search_recipe = []
        search_image = []
            
        for i in recipe_random_list:
            search_recipe.append(recipe_time.iloc[i]['title'])
            search_image.append(recipe_time.iloc[i]['Image_Name'])
        return search_recipe, search_image
    
    else:
        # filter recipe based on dietary preferences
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['vegetarian'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['peanut_free'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['gluten_free'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['low_calories'] == 1]
        if is_vegetarian:
            recipe_2 = recipe_2[recipe_2['kid_friendly'] == 1]
        
        
        if len(recipe_2) >= 3 :
            # create new dataframe (TF-IDF) of the ingredients    
            tfidf_recipe = pd.DataFrame(tfidf.fit_transform(recipe_2['ingr_parsed']).toarray(),
                                        columns = tfidf.get_feature_names_out(),
                                        index = recipe_2['title'])

            recipe_ingredient = tfidf_recipe.copy()
            
            # filter df based on the available ingredients
            try:
                for i in ingredient:
                    recipe_ingredient = recipe_ingredient[recipe_ingredient[i] > 0]
            except:
                st.error('You have entered an ingredient that does not match your dietary preferences. Please recheck your ingredient input.', icon="🚨")
                        
                
        # if there's more than 3 recipes with all the selected ingredients
        if (len(recipe_2) >= 3) & (len(recipe_ingredient) >= 3):
            # join back with original dataframe
            recipe_ingredient.reset_index(inplace = True)
            recipe_ingredient = recipe_ingredient[['title']]
            recipe_ingredient = recipe_ingredient.join(recipe.set_index('title'), on='title')

            # get scores
            ingr_tfidf = tfidf.transform(recipe_ingredient['ingr_parsed'])
            title_tfidf = tfidf.transform(recipe_ingredient['title_parsed'])
            i_vector = tfidf.transform([' '.join(ingredient)])
            recipe_scores = ingr_tfidf*i_vector.T*0.5 + title_tfidf*i_vector.T*0.5

            #sort scores & get the top 3 similarity scores, then get the recipe index
            sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending = False)[0:3].index

            # return
            search_recipe = []
            search_image = []

            for i in sorted_index:
                search_recipe.append(recipe_ingredient.iloc[i]['title'])
                search_image.append(recipe_ingredient.iloc[i]['Image_Name'])
            return search_recipe, search_image
  

        # if there's less than 3 recipes with all the selected ingredients
        else:
            # get scores
            ingr_tfidf = tfidf.transform(recipe_2['ingr_parsed'])
            title_tfidf = tfidf.transform(recipe_2['title_parsed'])

            i_vector = tfidf.transform([' '.join(ingredient)])
            recipe_scores = ingr_tfidf*i_vector.T*0.5 + title_tfidf*i_vector.T*0.5

            #sort scores & get the top 3 similarity scores, then get the recipe index
            sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending = False)[0:3].index

            # return
            search_recipe = []
            search_image = []

            for i in sorted_index:
                search_recipe.append(recipe_2.iloc[i]['title'])
                search_image.append(recipe_2.iloc[i]['Image_Name'])
            return search_recipe, search_image

def page_one():
    st.session_state.page = 1
    
def page_two(name):    
    st.session_state.page = 2
    st.session_state.recipe_name = name

def home_page():
    st.session_state.page = 0

        
#---------------IMPORT FILE---------------#
recipe_dict = pickle.load(open('recipes.pkl', 'rb'))
recipe = pd.DataFrame(recipe_dict)
recipe_1 = pd.DataFrame(recipe_dict)  

ingredient = pickle.load(open('ingredients.pkl', 'rb'))
ingredient = pd.DataFrame(ingredient)

#cosine_sim = pickle.load(open('recipe_recommendations.pkl','rb'))



#------------------------------------------------------#
#------------------------SIDEBAR-----------------------#
#------------------------------------------------------#
             
#---------------INGREDIENT SELECTION---------------#
with st.sidebar.form("my_form"):
    st.write("Select Your Ingredients")
    
    ingredientbox_1 = st.multiselect(
        "Pantry Essentials 🥚",
        ("Salt", "Olive oil", "Garlic", "Sugar", "Onion",
        "unsalted butter", "Egg", "Pepper", "Black pepper", "Lemon juice")
    )

    ingredientbox_2 = st.multiselect(
        "Vegetable 🥕",
        ("Arrowroot", "Asparagus", "Bamboo shoots", "Beetroot", "bell pepper", "Bok choi", "Broccoli", "Brussels sprouts", "Butterhead lettuce", "Butternut squash", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Celery", "Cherry tomatoes", "Chives", "corn", "Cucumber", "eggplant", "Green beans", "Iceberg lettuce", "Kale", "Leek", "Lettuce", "Mushroom", "Olive", "Paprika", "Parsnip", "Potato", "Pumpkin", "Red onion", "Shallot", "scallion", "Spinach", "Tomato", "Yam", "zucchini")
    )

    ingredientbox_3 = st.multiselect(
        "Meat & Poultry 🐓",
        ("Bacon", "Beef", "Chicken", "Chicken breast", "Chicken thigh", "Duck", "Ham", "Lamb", "Oxtail", "Pork", "Pork belly", "Pork chop", "Pork fillet", "Pulled pork", "Sausage", "Sirloin", "Turkey")
    )
    
    ingredientbox_4 = st.multiselect(
        "Seafood 🦀",
        ("Anchovies", "Clams", "Cockles", "Cod", "Crab", "Crayfish", "Cuttlefish", "Fish", "Halibut", "Mussels", "Octopus", "Oyster", "Prawn", "Salmon", "Sardine", "Scallop", "Seaweed", "Squid", "Fresh tuna")
    )

    ingredientbox_5 = st.multiselect(
        "All Ingredients 🧀",
        (ingredient['Ingredient_ori'].to_list())
    )
    
    
#-----------------DIET FILTERING-----------------#
    with st.expander("Dietary Preference 🙅🏻‍♀️"):
        is_vegetarian = st.checkbox('Vegetarian')
        is_peanut_free = st.checkbox('Peanut-Free') 
        is_gluten_free = st.checkbox('Gluten-Free')
        is_low_calories = st.checkbox('Low Calories')
        is_kid_friendly = st.checkbox('Kid-Friendly')
        
        
    search_recipe_button = st.form_submit_button("Search Recipe 🔍", on_click = page_one)
        
    ingredientbox = ingredientbox_1 + ingredientbox_2 + ingredientbox_3 + ingredientbox_4 + ingredientbox_5
    
    selected_ingredients = []
   
    for item in ingredientbox:
        item = ingredient[ingredient['Ingredient_ori'] == item]['ingredient'].values[0]
        
        if item not in selected_ingredients:
            selected_ingredients.append(item)


            
#------------------------------------------------------#
#--------------------LANDING PAGE----------------------#
#------------------------------------------------------#

if "page" not in st.session_state:
    st.session_state.page = 0
    

if st.session_state.page == 0:
    st.image('dataset/food_images/_kiwi_kitchen_4.jpg')
    st.header(f"Welcome to Kiwi Kitchen!")
    st.write(f"Explore new recipe ideas based on ingredients you have 🍅🍠🥓")
    st.write(f"Please select the ingredients on the sidebar and your dietary preference (if any)!")
    st.write(f"Click 'Search Recipe 🔍' once you are ready!")
    st.write(" ")
    st.write(" ")
    st.write(f'<p style="color:#ff8111;font-size:20px;">Get in touch with me!</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.write('Janet Jeun')
    with col2:
        if st.button('Phone'):
            st.markdown('📞+65 84041508')
    with col3:
        if st.button('Email'):
            st.markdown('<a href="mailto:jeunjanet@gmail.com">jeunjanet@gmail.com</a>', unsafe_allow_html=True)
    with col4:
        if st.button('Github'):
            webbrowser.open_new_tab('https://github.com/janetjeun')
    with col5:
        if st.button('LinkedIn'):
            webbrowser.open_new_tab('https://www.linkedin.com/in/janetjeun/')
    #with col6:
        #if st.button('Resume'):
            #webbrowser.open_new_tab('https://www.linkedin.com/in/janetjeun/')

    
#-----------------------------------------------------#    
#-----------------SEARCH RECIPE PAGE------------------#
#-----------------------------------------------------#

elif st.session_state.page == 1:

    search_recipe, search_image = search_recipes(selected_ingredients,
                                                 is_vegetarian, is_peanut_free, is_gluten_free, is_low_calories, is_kid_friendly)

    
    #-----------------FIRST RECIPE------------------#
    col1, col2 = st.columns([1,2])
    
    with col1:
        image_path = 'dataset/food_images/' + search_image[0] + '.jpg'
        st.image(image_path, use_column_width = True)
    with col2:
        st.write(f'<p style="color:#ff8111;font-size:20px;">{search_recipe[0]}</p>', unsafe_allow_html=True)
        st.button("View Full Recipe", key = 0, on_click = page_two, args = [search_recipe[0]])
    
    
    #-----------------SECOND RECIPE------------------#
    col1, col2 = st.columns([1,2])
    
    with col1:
        image_path = 'dataset/food_images/' + search_image[1] + '.jpg'
        st.image(image_path, use_column_width = True)
        
    with col2:      
        st.write(f'<p style="color:#ff8111;font-size:20px;">{search_recipe[1]}</p>', unsafe_allow_html=True)
        st.button("View Full Recipe", key = 1, on_click = page_two, args = [search_recipe[1]])
    
    
    #-----------------THIRD RECIPE------------------#
    col1, col2 = st.columns([1,2])
    
    with col1:
        image_path = 'dataset/food_images/' + search_image[2] + '.jpg'
        st.image(image_path, use_column_width = True)
        
    with col2:
        st.write(f'<p style="color:#ff8111;font-size:20px;">{search_recipe[2]}</p>', unsafe_allow_html=True)
        st.button("View Full Recipe", key = 2, on_click = page_two, args = [search_recipe[2]])
    
    st.write(" ")
    st.write(" ")
    
    st.button('Back to Home', on_click = home_page)
    
    
#------------------------------------------------------#
#-----------------RECIPE DETAILS PAGE------------------#
#------------------------------------------------------#

elif st.session_state.page == 2:
    
    #-----------------RECIPE DETAILS-----------------#
    recipe_details = st.session_state.recipe_name
    recipe_index = recipe[recipe['title'] == recipe_details].index[0]
    st.write(f'<p style="color:#ff8111;font-size:28px;">{recipe_details}</p>', unsafe_allow_html=True)

    image_path = 'dataset/food_images/' + recipe.iloc[recipe_index]['Image_Name'] + '.jpg'
    st.image(image_path)
    
    st.write(f'<p style="color:#ff8111;font-size:18px;">Ingredients:</p>', unsafe_allow_html=True)
    for item in recipe.iloc[recipe_index]['ingredients']:
        st.write(item)
        
    st.write(" ")
    
    st.write(f'<p style="color:#ff8111;font-size:18px;">Directions:</p>', unsafe_allow_html=True)
    for item in recipe.iloc[recipe_index]['directions']:
        st.write(item)
        
    #st.write(f'<p style="color:#ff8111;font-size:18px;">Directions:</p>', unsafe_allow_html=True)
    #for item in recipe.iloc[recipe_index]['tags']:
        #st.write(item)

    
    st.write(" ")
    st.write(" ")
    st.write(" ")
    
    #-----------------YOU MIGHT ALSO LIKE-----------------#
    
    st.write(f'<p style="color:#ff8111;font-size:24px;">You might also like ❤:</p>', unsafe_allow_html=True)
    similar_recipe_name, similar_recipe_image = get_recommendations(recipe_details,
                                                                    is_vegetarian, is_peanut_free, is_gluten_free,
                                                                    is_low_calories, is_kid_friendly)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        image_path = 'dataset/food_images/' + similar_recipe_image[0] + '.jpg'
        st.image(image_path, use_column_width = True)
        st.write(similar_recipe_name[0])
        st.button("View Full Recipe", key = 0, on_click = page_two, args = [similar_recipe_name[0]])
        
    with col2:
        image_path = 'dataset/food_images/' + similar_recipe_image[1] + '.jpg'
        st.image(image_path, use_column_width = True)
        st.write(similar_recipe_name[1])
        st.button("View Full Recipe", key = 1, on_click = page_two, args = [similar_recipe_name[1]])
        
    with col3:
        image_path = 'dataset/food_images/' + similar_recipe_image[2] + '.jpg'
        st.image(image_path, use_column_width = True)
        st.write(similar_recipe_name[2])
        st.button("View Full Recipe", key = 2, on_click = page_two, args = [similar_recipe_name[2]])
        
        
    st.write(" ")
    st.write(" ")
    st.write(" ")
    
    st.button('Back to Home', on_click = home_page)

    
        