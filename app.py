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
import openai.error  # Correctly import OpenAIError

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
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.8)

    excluded_ingredient = request.form.get('excluded_ingredient', '')
    substitute_ingredient = request.form.get('substitute_ingredient', '')

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
        Specify the meal preference: breakfast, lunch, dinner, or snack.
    """

    topic = recipe_name or meal_preference or "recipe"
    prompt_text = template.format(excluded_ingredient, substitute_ingredient, cooking_time or 'any')
    prompt_template = PromptTemplate(input_variables=["recipe_name"], template=prompt_text)

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
    )

    response = chain({"recipe_name": topic})['text']
    recipe_components = response.split("\n\n")

    recipe_name = recipe_components[0].replace("1. Recipe Name:", "").strip()
    ingredients_list = [ingredient.strip() for ingredient in recipe_components[1].replace("2. List of ingredients:", "").strip().split("\n")]
    recipe_steps = "\n".join([step.strip() for step in recipe_components[2].replace("3. Recipe steps:", "").strip().split("\n")])
    cooking_time = recipe_components[3].replace("4. Cooking Time:", "").strip()
    meal_preference = recipe_components[4].replace("5. Meal Preferences:", "").strip()

    recipe_content = {
        "recipe_name": recipe_name,
        "cooking_time": cooking_time,
        "ingredients": ingredients_list,
        "recipe_steps": recipe_steps,
        "meal_preference": meal_preference,
    }

    return recipe_content

def is_recipe_related(name):
    keywords = ['cake', 'salad', 'soup', 'curry', 'bread', 'pasta', 'pizza', 'dish', 'meal', 'snack']
    return any(keyword in name.lower() for keyword in keywords)

def generate_image(recipe_name):
    if not is_recipe_related(recipe_name):
        logging.warning(f"Invalid recipe name for image generation: {recipe_name}")
        return None

    try:
        prompt = (
            f"Generate a high-quality image of a cooked dish titled '{recipe_name}'. "
            "Focus exclusively on the plated food presentation in a clean and minimalistic style. "
            "Avoid including any unrelated visuals such as people, text, or background elements. "
            "The image should capture the essence of the dish with vibrant colors and professional styling."
        )

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )

        return response['data'][0]['url']
    except openai.error.OpenAIError as e:
        logging.error(f"Image generation error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    recipe_ingredients = request.form['recipe_ingredients']
    cooking_time = request.form['cooking_time']
    meal_preference = request.form['meal_preference']

    recipe_name, *ingredients_list = recipe_ingredients.split('\n')
    ingredients_list = [ingredient.strip() for ingredient in ingredients_list if ingredient.strip()]

    user_interaction_logger.info(f"Generating recipe with ingredients: {recipe_ingredients}, cooking time: {cooking_time}, meal preference: {meal_preference}")

    recipe_content = search_recipe(recipe_name=recipe_name, ingredients=ingredients_list, meal_preference=meal_preference)
    image_url = generate_image(recipe_name)

    most_searched_keywords_logger.info(recipe_name)

    return render_template('view.html', recipe_content=recipe_content, image_url=image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
