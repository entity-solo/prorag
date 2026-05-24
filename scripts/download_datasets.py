import os
import json
import requests
import sys

DATA_DIR = "data"
RESULTS_DIR = "results"
HOTPOT_URL = "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json"
FILE_PATH = os.path.join(DATA_DIR, "hotpot_dev_distractor_v1.json")

# 5 real-world QA samples as fallback in case CMU/S3 download fails
FALLBACK_SAMPLES = [
    {
        "_id": "fb_01",
        "question": "Were Einstein and Newton both physicists?",
        "answer": "yes",
        "supporting_facts": [
            ["Albert Einstein", 0],
            ["Isaac Newton", 0]
        ],
        "context": [
            ["Albert Einstein", ["Albert Einstein was a German-born theoretical physicist.", "He developed the theory of relativity."]],
            ["Isaac Newton", ["Isaac Newton was an English mathematician and physicist.", "He is widely recognized as one of the most influential scientists of all time."]],
            ["Thomas Edison", ["Thomas Edison was an American inventor.", "He developed many devices in fields such as electric power generation."]]
        ]
    },
    {
        "_id": "fb_02",
        "question": "What theory did Albert Einstein develop in 1905?",
        "answer": "theory of relativity",
        "supporting_facts": [
            ["Albert Einstein", 1]
        ],
        "context": [
            ["Albert Einstein", ["Albert Einstein was a physicist.", "He developed the theory of relativity in 1905."]],
            ["Max Planck", ["Max Planck was a German theoretical physicist.", "He originated quantum theory."]]
        ]
    },
    {
        "_id": "fb_03",
        "question": "At what temperature does water boil at standard pressure?",
        "answer": "100 degrees",
        "supporting_facts": [
            ["Water Properties", 0]
        ],
        "context": [
            ["Water Properties", ["Water boils at 100 degrees Celsius at standard atmospheric pressure.", "Water freezes at 0 degrees Celsius."]],
            ["Ethanol Properties", ["Ethanol boils at 78.37 degrees Celsius.", "It is a clear, colorless liquid."]]
        ]
    },
    {
        "_id": "fb_04",
        "question": "Where is the Eiffel Tower located?",
        "answer": "Paris",
        "supporting_facts": [
            ["Eiffel Tower", 1]
        ],
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars.", "It is located in Paris, France.", "It was constructed from 1887 to 1889."]],
            ["Colosseum", ["The Colosseum is an oval amphitheatre.", "It is located in the centre of the city of Rome, Italy."]]
        ]
    },
    {
        "_id": "fb_05",
        "question": "Do scientific studies show that vaccines cause autism?",
        "answer": "no",
        "supporting_facts": [
            ["Vaccine Safety", 0]
        ],
        "context": [
            ["Vaccine Safety", ["Multiple rigorous scientific studies have shown that vaccines do not cause autism.", "The CDC states there is no link between vaccines and autism."]],
            ["Andrew Wakefield", ["Andrew Wakefield is a struck-off British physician.", "He published a fraudulent paper in 1998 linking the MMR vaccine to autism."]]
        ]
    }
]


def create_directories():
    for d in [DATA_DIR, RESULTS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")


def write_fallback():
    print("Writing fallback sample dataset...")
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(FALLBACK_SAMPLES, f, ensure_ascii=False, indent=2)
    print(f"Fallback dataset created at {FILE_PATH} (contains {len(FALLBACK_SAMPLES)} sample questions).")


def download_hotpot():
    create_directories()
    
    if os.path.exists(FILE_PATH):
        print(f"Dataset already exists at {FILE_PATH}")
        return

    print(f"Downloading HotpotQA dev set from {HOTPOT_URL}...")
    print("Note: This file is ~45MB. If you want to skip or cancel, press Ctrl+C to use the local fallback.")
    
    try:
        response = requests.get(HOTPOT_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(FILE_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\rDownloaded: {downloaded // 1024} KB")
                        sys.stdout.flush()
        print("\nDownload complete!")
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
        write_fallback()
    except Exception as e:
        print(f"\nDownload failed: {e}")
        write_fallback()


if __name__ == "__main__":
    download_hotpot()
