import openai

# Replace 'your-api-key' with your actual API key
openai.api_key = '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'

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

    except openai.error.InvalidRequestError as e:
        print(f"Invalid request: {e.user_message}")
    except openai.error.APIConnectionError:
        print("Error connecting to the API. Please try again later.")
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please wait before making more requests.")
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")

    return None

if __name__ == "__main__":
    prompt = "pizza"  # Optimized prompt
    image_url = generate_image(prompt)

    if image_url:
        print(f"Generated image URL: {image_url}")