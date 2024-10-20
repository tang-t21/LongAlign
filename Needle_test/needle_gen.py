import random
import string
import argparse
import json

def generate_unique_secret_keys(num_keys=1000, key_length=4):
    unique_keys = set()
    characters = string.ascii_letters + string.digits  # Include letters and digits
    while len(unique_keys) < num_keys:
        key = ''.join(random.choices(characters, k=key_length))
        unique_keys.add(f" {key} is one of the secret keys.")
    
    return list(unique_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_keys", type=int, default=1000, help="Number of unique secret keys to generate.")
    parser.add_argument("--key_length", type=int, default=4, help="Length of the secret keys.")
    parser.add_argument('--output_dir', type=str, default="./needles", help="Output file to save the generated secret keys.")
    args = parser.parse_args()
    # Generate 1000 unique secret keys with a length of 5 tokens
    unique_secret_keys = generate_unique_secret_keys(num_keys=args.num_keys, key_length=args.key_length)

    # Format the output with the question and generated unique needles
    output_unique_secret_keys = {
        "needles": unique_secret_keys,
        "retrieval_question": "What are all secret keys in the above context? They are in the form of '<key> is one of the secret keys'. Find all of them."
    }

    # Save the generated secret keys to a file
    with open(f"{args.output_dir}/{args.num_keys}keys_len_{args.key_length}.json", "w") as f:
        json.dump(output_unique_secret_keys, f, indent=4)
    