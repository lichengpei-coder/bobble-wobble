import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="AIGC Detection Inference Script")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--output_json", type=str, default="predictions.json", help="Output path for JSON results")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    results = []

    # TODO: Load your trained PyTorch model here
    
    # Iterate over image files
    for img_path in image_dir.glob("*.[jJ][pP][gG]"):
        # Placeholder prediction score (0.0 = Real, 1.0 = Fake)
        dummy_score = 0.5 
        
        results.append({
            "image_path": str(img_path),
            "pred": round(dummy_score, 4)
        })

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Predictions successfully saved to {args.output_json}")

if __name__ == "__main__":
    main()
