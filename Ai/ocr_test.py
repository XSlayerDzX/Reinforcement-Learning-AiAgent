# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your local server
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # Local server address
    api_key="obQog4mAaBRuPZZBIoti"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="clashroyalbot-z9idj",
    workflow_id="custom-workflow-3",
    images={
        "image": r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\models\temp_screens\capture_43.png" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)


