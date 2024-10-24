from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from openai_api_key import openai_api_key  # Assuming you have an openai_api_key.py file with your API key
import logging
import re
from datetime import datetime, timedelta
import openai

app = Flask(__name__)
openai.api_key = openai_api_key
# Set up logging
log_file_name = 'user_activity.log'
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add a StreamHandler to log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Set up logging for most searched keywords
most_searched_keywords_file = 'most_searched_keywords.log'
most_searched_keywords_handler = logging.FileHandler(most_searched_keywords_file)
most_searched_keywords_handler.setLevel(logging.INFO)
most_searched_keywords_handler.setFormatter(formatter)
most_searched_keywords_logger = logging.getLogger('most_searched_keywords')
most_searched_keywords_logger.addHandler(most_searched_keywords_handler)

# Set up logging for user interactions
user_interaction_file = 'user_interaction.log'
user_interaction_handler = logging.FileHandler(user_interaction_file)
user_interaction_handler.setLevel(logging.INFO)
user_interaction_handler.setFormatter(formatter)
user_interaction_logger = logging.getLogger('user_interaction')
user_interaction_logger.addHandler(user_interaction_handler)

@app.errorhandler(400)
def handle_bad_request(e):
    return render_template('Errors/400.html'), 400

@app.errorhandler(401)
def handle_unauthorized(e):
    return render_template('Errors/401.html'), 401

@app.errorhandler(403)
def handle_forbidden(e):
    return render_template('Errors/403.html'), 403

@app.errorhandler(404)
def handle_not_found(e):
    return render_template('Errors/404.html'), 404

@app.errorhandler(405)
def handle_method_not_allowed(e):
    return render_template('Errors/405.html'), 405

@app.errorhandler(429)
def handle_too_many_requests(e):
    return render_template('Errors/429.html'), 429

@app.errorhandler(500)
def handle_internal_server_error(e):
    return render_template('Errors/500.html'), 500

def search_recipe(recipe_name=None, ingredients=None, meal_preference=None, cooking_time=None):
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.8)

    # Get excluded and substitute ingredients from the form (if any)
    excluded_ingredient = request.form.get('excluded_ingredient', '')
    substitute_ingredient = request.form.get('substitute_ingredient', '')

    # Start building the template for the recipe prompt
    template = """
    Your task is to generate a recipe based on the provided recipe name and ingredients. 
    Exclude the ingredient '{}' and use '{}' instead. Keep the cooking time around '{}' minutes (within 5 minutes).


    1. Recipe Name:
        Provide a clear and concise title for the dish.

    2. List of ingredients:
        Please list all the ingredients needed for the recipe, each on a new line.

    3. Recipe steps:
        Provide step-by-step instructions on how to prepare the dish. Start each step with a number or bullet point, each on a new line.

    4. Cooking Time:
        Provide an estimated cooking time for the recipe in minutes.

    5. Meal Preferences:
        Specify the meal preference: breakfast, lunch, dinner, or snack (choose only one of these four).
   
    **Image Generation Instructions**:
    - Generate a visually appealing image that represents the final dish, focusing exclusively on the recipe content. 
    - Ensure the image captures the essence of the dish, avoiding unrelated visuals or subjects.

    Feel free to include any additional tips or details to make the recipe even more delicious.

    Topic: {}
    """

    # Determine the topic for the recipe prompt
    if meal_preference:
        topic = f"{meal_preference} recipe"
    elif recipe_name:
        topic = recipe_name
    elif ingredients:
        topic = "recipe with " + ", ".join(ingredients)
    else:
        raise ValueError("Please provide either recipe_name, ingredients, or meal_preference.")

    # Modify the ingredient list by excluding and substituting ingredients if applicable
    if excluded_ingredient and ingredients and excluded_ingredient in ingredients:
        ingredients.remove(excluded_ingredient)
        if substitute_ingredient:
            ingredients.append(substitute_ingredient)

    # Incorporate cooking time if provided
    if cooking_time:
        topic += f" around {cooking_time} minutes"
    else:
        cooking_time = 'any'

    # Finalize the prompt text
    prompt_text = template.format(excluded_ingredient, substitute_ingredient, cooking_time, topic)

    # Create the prompt template
    prompt_template = PromptTemplate(input_variables=["recipe_name"], template=prompt_text)

    # Create the LLM chain for generating the response
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
    )

    # Get the response by running the chain
    response = chain({"recipe_name": topic})['text']

    # Split the response into the recipe components
    recipe_components = response.split("\n\n")

    # Extract recipe details
    recipe_name = recipe_components[0].replace("1. Recipe Name:", "").strip()
    ingredients_list = [ingredient.strip() for ingredient in recipe_components[1].replace("2. List of ingredients:", "").strip().split("\n")]
    recipe_steps = "\n".join([step.strip() for step in recipe_components[2].replace("3. Recipe steps:", "").strip().split("\n")])
    cooking_time = recipe_components[3].replace("4. Cooking Time:", "").strip()
    meal_preference = recipe_components[4].replace("5. Meal Preferences:", "").strip()
    additional_info = recipe_components[5].replace("Additional tip:", "").strip() if len(recipe_components) > 5 else None

    # Construct the recipe dictionary
    recipe_content = {
        "recipe_name": recipe_name,
        "cooking_time": cooking_time,
        "ingredients": ingredients_list,
        "recipe_steps": recipe_steps,
        "meal_preference": meal_preference,
        "additional_info": additional_info,
    }

    return recipe_content


def generate_image(prompt):
    try:
        # Make a request to generate an image with DALL·E
        response = openai.Image.create(
            prompt=prompt,  # Optimized prompt
            n=1,            # Number of images to generate
            size="1024x1024"  # Size of the image
        )

        # Access and return the generated image URL
        image_url = response['data'][0]['url']
        return image_url
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    recipe_ingredients = request.form['recipe_ingredients']
    cooking_time = request.form['cooking_time']
    meal_preference = request.form['meal_preference']

    # Split the recipe ingredients into recipe name and ingredients
    recipe_name, *ingredients_list = recipe_ingredients.split('\n')

    # Join the remaining lines as ingredients list
    ingredients_list = [ingredient.strip() for ingredient in ingredients_list if ingredient.strip()]

    # Log user activity
    user_interaction_logger.info(f"Generating recipe with ingredients: {recipe_ingredients}, cooking time: {cooking_time}, meal preference: {meal_preference}")

    recipe_content = search_recipe(recipe_name=recipe_name, ingredients=ingredients_list, meal_preference=meal_preference)

    # Generate image using the recipe name
    image_url = generate_image(recipe_name)

    # Log most searched keywords
    most_searched_keywords_logger.info(recipe_name)

    return render_template('view.html', recipe_content=recipe_content, image_url=image_url)

@app.route('/get_substitutions', methods=['POST'])
def get_substitutions():
    ingredient = request.json['ingredient']
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    template = f"Given the ingredient '{ingredient}', Based on the following ingredient list, please suggest five substitutions for {ingredient} and only name"
    response = llm.invoke(template).content
    substitutions = response.split("\n")
    return jsonify(substitutions)

@app.route('/Recommandation')
def recommandation_page():
    # Initialize recipe_counts dictionary
    recipe_counts = {}

    # Open the log file and read its contents
    with open('most_searched_keywords.log', 'r') as file:
        file_data = file.readlines()

        # Define patterns for date, time, and recipe
        date_pattern = r"(\d{4}-\d{2}-\d{2})"
        time_pattern = r"(\d{2}:\d{2}:\d{2})"
        recipe_pattern = r" - INFO - (.*)"

        for line in file_data:
            # Extract date
            date_match = re.search(date_pattern, line)
            if date_match:
                date = date_match.group(1)

                # Parse date to datetime object
                log_date = datetime.strptime(date, '%Y-%m-%d')

                # Check if log entry is within the last 48 hours
                if log_date >= (datetime.now() - timedelta(hours=48)):

                    # Extract recipe
                    recipe_match = re.search(recipe_pattern, line)
                    if recipe_match:
                        recipe_name = recipe_match.group(1)

                        # Increment count for the recipe
                        if recipe_name in recipe_counts:
                            recipe_counts[recipe_name] += 1
                        else:
                            recipe_counts[recipe_name] = 1

    # Sort recipes by frequency
    most_searched_recipes = sorted(recipe_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the top 5 recipes
    top_5_recipes = most_searched_recipes[:5]

    return render_template('Recommandation.html', top_5_recipes=top_5_recipes)

if __name__ == '__main__':
     #app.run(debug=True)
     app.run(host='0.0.0.0', port=8080)
