import openai
import sys

if len(sys.argv) < 2:
    print("Please provide the OpenAI API key as a command-line argument.")
else:
    openai.api_key = sys.argv[1]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        )
        print(response.choices[0].message['content'])
    except Exception as e:
        print(f"An error occurred: {e}")
