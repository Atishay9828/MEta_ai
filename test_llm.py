import os
from openai import OpenAI
import sys

def main():
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not api_base_url or not model_name or not hf_token:
        print("ERROR: Missing one or more environment variables. Ensure API_BASE_URL, MODEL_NAME, and HF_TOKEN are set.")
        sys.exit(1)

    print("Status: Environment variables confirmed securely retrieved.")

    try:
        client = OpenAI(
            base_url=api_base_url,
            api_key=hf_token
        )

        print(f"Testing connectivity to model: {model_name}...")

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=20
        )

        print("\n=== Model Response ===")
        print(response.choices[0].message.content.strip())
        print("======================\n")
        print("Success: Connectivity verified securely.")

    except Exception as e:
        print(f"Error encountered during API call: {e}")

if __name__ == "__main__":
    main()
