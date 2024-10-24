import re
from flask import Flask, render_template
from datetime import datetime, timedelta

app = Flask(__name__)


@app.route('/')
def Recommandation():
    with open('user_activity.log', 'r') as file:
        file_data = file.readlines()
        start_time = datetime.now() - timedelta(hours=48)

        date_pattern = r"(\d{4}-\d{2}-\d{2})"
        time_pattern = r"(\d{2}:\d{2}:\d{2})"
        recipe_pattern = r"Generating recipe with ingredients: (.*?),"

        recipe_counts = {}

        for line in file_data:
            # Extract date
            date_match = re.search(date_pattern, line)
            if date_match:
                date = date_match.group(1)

            # Extract time
            time_match = re.search(time_pattern, line)
            if time_match:
                time = time_match.group(1)

            # Extract recipe
            recipe_match = re.search(recipe_pattern, line)
            if recipe_match:
                recipe = recipe_match.group(1)

                # Update recipe count
                if recipe.strip():  # Check if recipe name is not blank
                    if recipe in recipe_counts:
                        recipe_counts[recipe] += 1
                    else:
                        recipe_counts[recipe] = 1

    # Sort recipes by count in descending order
    sorted_recipes = sorted(recipe_counts.items(), key=lambda x: x[1], reverse=True)

    return render_template('Recommandation.html', sorted_recipes=sorted_recipes)


if __name__ == '__main__':
    app.run(debug=True)
