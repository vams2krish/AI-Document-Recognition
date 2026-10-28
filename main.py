import ollama

res = ollama.chat(
    model = 'llava:7b',
    messages = [
        {
            'role' : 'user',
            'content' : 'Extract the text in the image and what are the coordination ambiguity in the text',
            'images' : ['./document.png']
        }
    ]
)

print(res['message']['content'])