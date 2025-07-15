#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide the OpenAI API key as an argument."
  exit 1
fi

API_KEY=$1

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
